import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import sys

import cv2
import numpy as np
import tyro
import zmq
from loguru import logger
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat,
    AlignFilter, OBStreamType
)
from scipy.spatial.transform import Rotation as R

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

logger.remove()
logger.add(
    sys.stderr,
    format="<light-blue>{time:YYYY-MM-DD HH:mm:ss}</light-blue> | <level>{level: <8}</level> | <yellow>{name}</yellow>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

MIN_DEPTH = 200  # 200mm
MAX_DEPTH = 2000  # 2000mm


class TemporalFilter:
    """时域滤波器，平滑深度图"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


def produce_frames(queue):
    """相机数据采集进程"""
    pipeline = Pipeline()
    config = Config()
    temporal_filter = TemporalFilter(alpha=0.3)

    logger.info("正在初始化奥比中光相机...")

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None, "未找到深度相机"

        depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
        config.enable_stream(depth_profile)

        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        assert color_profiles is not None, "未找到彩色相机"

        color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        config.enable_stream(color_profile)

    except Exception as e:
        logger.warning(f"配置失败：{e}")
        logger.warning("尝试使用默认配置...")
        config.enable_all_stream()

    pipeline.start(config)

    # 💡 新增：向奥比中光 SDK 请求真实相机内参
    try:
        camera_param = pipeline.get_camera_param()
        # 注意：因为使用了 AlignFilter 对齐到了彩色图像，所以必须用彩色相机的内参！
        intrinsic = camera_param.rgb_intrinsic
        intrinsics = (intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy)
        logger.info(
            f"✅ 成功获取奥比中光真实内参: fx={intrinsic.fx:.1f}, fy={intrinsic.fy:.1f}, cx={intrinsic.cx:.1f}, cy={intrinsic.cy:.1f}")
    except Exception as e:
        logger.warning(f"⚠️ 获取真实内参失败，使用备用经验值: {e}")
        intrinsics = (500.0, 500.0, 320.0, 240.0)  # 640x480 的近似值

    align_filter = AlignFilter(OBStreamType.COLOR_STREAM)

    last_print_time = time.time()
    frame_count = 0

    logger.info("相机已启动，开始采集数据...")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                continue

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            try:
                c_v_frame = color_frame.as_video_frame()
                raw_color_data = np.asanyarray(c_v_frame.get_data(), dtype=np.uint8)
                raw_color_data = np.ascontiguousarray(raw_color_data)

                fmt = c_v_frame.get_format()

                if fmt == OBFormat.BGR:
                    color_img = raw_color_data.reshape((c_v_frame.get_height(), c_v_frame.get_width(), 3))
                elif fmt == OBFormat.RGB:
                    color_img = raw_color_data.reshape((c_v_frame.get_height(), c_v_frame.get_width(), 3))
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                else:
                    color_img = cv2.imdecode(raw_color_data, cv2.IMREAD_COLOR)

                if color_img is None:
                    continue

                d_v_frame = depth_frame.as_video_frame()
                width = d_v_frame.get_width()
                height = d_v_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_data = np.frombuffer(d_v_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale

                depth_data = np.where(
                    (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH),
                    depth_data,
                    0
                )

                depth_data = temporal_filter.process(depth_data.astype(np.uint16))

                if color_img.shape[:2] != depth_data.shape[:2]:
                    color_img = cv2.resize(color_img, (depth_data.shape[1], depth_data.shape[0]))

            except Exception as e:
                logger.error(f"数据转换异常：{e}")
                continue

            if queue.full():
                try:
                    queue.get_nowait()
                except:
                    pass
            queue.put((color_img, depth_data, intrinsics))

            frame_count += 1
            current_time = time.time()

            if current_time - last_print_time >= 2.0:
                elapsed = current_time - last_print_time
                fps = frame_count / elapsed
                logger.info(f"相机帧率：{fps:.1f} FPS")
                frame_count = 0
                last_print_time = current_time

    except KeyboardInterrupt:
        logger.info("\n停止相机采集...")
    except Exception as e:
        logger.error(f"相机异常：{e}")
    finally:
        pipeline.stop()


def start_vision_server(queue, robot_dir: str = None, config_path: str = None):
    """视觉服务端 - 支持 ZMQ 通信"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://0.0.0.0:5555")

    # 加载重定向配置
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type)

    # 状态跟踪
    prev_T = None
    frame_count = 0

    logger.info(f"视觉服务端启动：Dexterous 模式")
    logger.info(f"  重定向类型：{retargeting.optimizer.retargeting_type}")
    logger.info(f"  关节数量：{len(retargeting.joint_names)}")
    logger.info(f"  关节名：{retargeting.joint_names}")

    while True:
        try:
            color_img, depth_img, intrinsics = queue.get(timeout=2)
            while not queue.empty():
                color_img, depth_img, intrinsics = queue.get_nowait()
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            fx, fy, cx, cy = intrinsics
        except Empty:
            continue

        ts = int(time.time() * 1000)
        num_box, joint_pos, keypoint_2d, wrist_rot = detector.detect(rgb, ts)

        frame_count += 1

        if joint_pos is not None:
            # 计算腕部位姿
            h, w = depth_img.shape
            u, v = int(keypoint_2d[0].x * w), int(keypoint_2d[0].y * h)
            u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)

            z_mm = float(depth_img[v, u])
            z = z_mm * 0.001
            z = np.clip(z, 0.3, 2.0)

            # 🚨 修改 2：使用真实的出厂标定内参计算真实的 X, Y 米数！
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            T_curr = np.eye(4)
            T_curr[:3, :3] = wrist_rot

            # 填入最精确的物理坐标！
            T_curr[:3, 3] = [x, y, z]
            # 重定向计算
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "POSITION":
                ref_value = np.array(joint_pos[indices, :])
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            if ref_value.ndim == 1:
                ref_value = ref_value[None, :]
            elif ref_value.ndim == 0:
                logger.error("ref_value 计算异常，请检查 YAML 配置文件中的索引")
                continue

            qpos = retargeting.retarget(ref_value)
            logger.info(f"{hand_type} hand retargeting: {qpos}")

            if qpos is not None:
                socket.send_json({
                    "wrist_pose": T_curr.tolist(),
                    "robot_joints": qpos.tolist()
                })

                if frame_count % 30 == 0:
                    logger.debug(f"灵巧手关节：{[f'{x:.3f}' for x in qpos]}")
            else:
                socket.send_json({
                    "wrist_pose": T_curr.tolist(),
                    "robot_joints": [0.0] * len(retargeting.joint_names)
                })

            # 绘制可视化
            color_img = detector.draw_skeleton_on_image(color_img, keypoint_2d)

            # 计算捏合距离（用于显示）
            thumb_tip = joint_pos[4]
            index_tip = joint_pos[8]
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            gripper_val = np.clip((pinch_dist - 0.02) / (0.15 - 0.02) * 2.0 - 1.0, -1.0, 1.0)

            info_text = f"Dist={pinch_dist:.3f}, Grip={gripper_val:.2f}"
            cv2.putText(color_img, info_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mode_text = f"Mode: Dexterous"
            cv2.putText(color_img, mode_text, (10, color_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            socket.send_json({
                "wrist_pose": np.eye(4).tolist(),
                "robot_joints": [0.0] * 12
            })
            cv2.putText(color_img, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Teleop Server", color_img)
        if cv2.waitKey(1) == ord('q'):
            break


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """主函数 - 参照官方示例风格"""
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    logger.info(f"启动视觉服务端...")
    logger.info(f"  机器人：{robot_name.value}")
    logger.info(f"  重定向类型：{retargeting_type.value}")
    logger.info(f"  手性：{hand_type.value}")
    logger.info(f"  配置文件：{config_path}")

    q = multiprocessing.Queue(maxsize=2)
    producer = multiprocessing.Process(target=produce_frames, args=(q,))
    consumer = multiprocessing.Process(target=start_vision_server, args=(q, str(robot_dir), str(config_path)))

    producer.start()
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        logger.info("\n正在退出...")
        producer.terminate()
        consumer.terminate()


if __name__ == "__main__":
    tyro.cli(main)
