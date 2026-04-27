import multiprocessing
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Literal

import cv2
import numpy as np
import tyro
import zmq
from loguru import logger
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat,
    AlignFilter, OBStreamType
)

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dual_hand_detector import DualHandDetector
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

logger.remove()
logger.add(
    sys.stderr,
    format="<light-blue>{time:YYYY-MM-DD HH:mm:ss}</light-blue> | <level>{level: <8}</level> |"
           " <yellow>{name}</yellow>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

MIN_DEPTH = 20  # 20mm
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
    """realsense相机数据采集进程"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 获取相机内参
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            depth_img = np.asanyarray(aligned.get_depth_frame().get_data())
            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass
            queue.put((color_img, depth_img, (fx, fy, cx, cy)))
    finally:
        pipeline.stop()


# def produce_frames(queue):
#     """奥比中光相机数据采集进程"""
#     pipeline = Pipeline()
#     config = Config()
#     temporal_filter = TemporalFilter(alpha=0.3)
#
#     logger.info("正在初始化奥比中光相机...")
#
#     try:
#         # 配置深度流
#         profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#         depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
#         config.enable_stream(depth_profile)
#
#         # 配置彩色流
#         color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
#         config.enable_stream(color_profile)
#
#     except Exception as e:
#         logger.warning(f"⚠️ 配置失败：{e}，尝试默认配置")
#         config.enable_all_stream()
#
#     pipeline.start(config)
#     align_filter = AlignFilter(OBStreamType.COLOR_STREAM)
#
#     camera_param = pipeline.get_camera_param()
#     intrinsic = camera_param.rgb_intrinsic
#     intrinsics = (intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy)
#     logger.info(
#         f"✅ 成功获取奥比中光真实内参: fx={intrinsic.fx:.1f}, fy={intrinsic.fy:.1f}, cx={intrinsic.cx:.1f}, cy={intrinsic.cy:.1f}")
#
#     last_print_time = time.time()
#     frame_count = 0
#
#     logger.info("相机已启动，开始采集数据...")
#
#     try:
#         while True:
#             frames = pipeline.wait_for_frames(1000)
#             if frames is None:
#                 continue
#
#             aligned_frames = align_filter.process(frames)
#             if aligned_frames is None:
#                 continue
#
#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()
#
#             if color_frame is None or depth_frame is None:
#                 continue
#
#             try:
#                 c_v_frame = color_frame.as_video_frame()
#                 raw_color_data = np.asanyarray(c_v_frame.get_data(), dtype=np.uint8)
#                 raw_color_data = np.ascontiguousarray(raw_color_data)
#
#                 fmt = c_v_frame.get_format()
#
#                 if fmt == OBFormat.BGR:
#                     color_img = raw_color_data.reshape((c_v_frame.get_height(), c_v_frame.get_width(), 3))
#                 elif fmt == OBFormat.RGB:
#                     color_img = raw_color_data.reshape((c_v_frame.get_height(), c_v_frame.get_width(), 3))
#                     color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
#                 else:
#                     color_img = cv2.imdecode(raw_color_data, cv2.IMREAD_COLOR)
#
#                 if color_img is None:
#                     continue
#
#                 d_v_frame = depth_frame.as_video_frame()
#                 width = d_v_frame.get_width()
#                 height = d_v_frame.get_height()
#                 scale = depth_frame.get_depth_scale()
#
#                 depth_data = np.frombuffer(d_v_frame.get_data(), dtype=np.uint16)
#                 depth_data = depth_data.reshape((height, width))
#
#                 depth_data = depth_data.astype(np.float32) * scale
#
#                 depth_data = np.where(
#                     (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH),
#                     depth_data,
#                     0
#                 )
#
#                 depth_data = temporal_filter.process(depth_data.astype(np.uint16))
#
#             except Exception as e:
#                 logger.error(f"数据转换异常：{e}")
#                 continue
#
#             if queue.full():
#                 try:
#                     queue.get_nowait()
#                 except:
#                     pass
#             queue.put((color_img, depth_data, intrinsics))
#
#             frame_count += 1
#             current_time = time.time()
#
#             if current_time - last_print_time >= 2.0:
#                 elapsed = current_time - last_print_time
#                 fps = frame_count / elapsed
#                 logger.info(f"相机帧率：{fps:.1f} FPS")
#                 frame_count = 0
#                 last_print_time = current_time
#
#     except KeyboardInterrupt:
#         logger.info("\n停止相机采集...")
#     except Exception as e:
#         logger.error(f"相机异常：{e}")
#     finally:
#         pipeline.stop()


def start_vision_server(queue, robot_dir: str, config_paths: dict):
    """视觉服务端 - 支持双臂/单臂 ZMQ 通信"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://0.0.0.0:7777")

    # 加载重定向配置
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting_dict = {}

    for ht, config_path in config_paths.items():
        logger.info(f"Loading {ht} hand retargeting config: {config_path}")
        retargeting_dict[ht] = RetargetingConfig.load_from_file(config_path).build()
        logger.info(f"  {ht} 关节数量: {len(retargeting_dict[ht].joint_names)}")
        logger.info(f"  {ht} 关节映射关系: {retargeting_dict[ht].joint_names}")

    # MediaPipe: selfie=False 以确保左右手不反转
    detector = DualHandDetector(selfie=False)

    last_state = {}
    for ht, retargeting in retargeting_dict.items():
        last_state[ht] = {
            "wrist_pose": np.eye(4).tolist(),
            "robot_joints": [0.0] * len(retargeting.joint_names)
        }

    frame_count = 0
    logger.info(f"视觉服务端启动：双臂/单臂混合兼容模式")

    while True:
        try:
            color_img, depth_img, intrinsics = queue.get(timeout=2)
            while not queue.empty():
                color_img, depth_img, intrinsics = queue.get_nowait()
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            fx, fy, cx, cy = intrinsics
        except Empty:
            continue

        # 使用 DualHandDetector 预测
        _, joint_pos_dict, keypoint_2d_dict, wrist_rot_dict = detector.detect(rgb)

        # 过滤掉未被请求的手部数据
        for ht in list(joint_pos_dict.keys()):
            if ht not in retargeting_dict:
                joint_pos_dict.pop(ht, None)
                keypoint_2d_dict.pop(ht, None)
                wrist_rot_dict.pop(ht, None)

        frame_count += 1
        msg = {}

        if len(joint_pos_dict) > 0:
            for ht, joint_pos in joint_pos_dict.items():
                keypoint_2d = keypoint_2d_dict[ht]
                wrist_rot = wrist_rot_dict[ht]
                retargeting = retargeting_dict[ht]

                h, w = depth_img.shape
                u = int(keypoint_2d.landmark[0].x * w)
                v = int(keypoint_2d.landmark[0].y * h)
                u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)

                half_win = 2  # 5x5 窗口
                v_min, v_max = max(0, v - half_win), min(h, v + half_win + 1)
                u_min, u_max = max(0, u - half_win), min(w, u + half_win + 1)

                depth_roi = depth_img[v_min:v_max, u_min:u_max]
                valid_depths = depth_roi[depth_roi > 0]

                if len(valid_depths) > 0:
                    z_mm = float(np.median(valid_depths))
                else:
                    z_mm = 0

                z = z_mm * 0.001
                z = np.clip(z, 0.01, 2.0)

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                T_curr = np.eye(4)
                T_curr[:3, :3] = wrist_rot
                T_curr[:3, 3] = [x, y, z]

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

                qpos = retargeting.retarget(ref_value)

                # 记录该手最新的有效状态
                last_state[ht] = {
                    "wrist_pose": T_curr.tolist(),
                    "robot_joints": qpos.tolist() if qpos is not None else [0.0] * len(retargeting.joint_names)
                }

                # 右手绿色，左手黄色，位置上下错开
                y_offset = 30 if ht == "Right" else 90
                color = (0, 255, 0) if ht == "Right" else (255, 255, 0)

                # 1. 显示手腕 3D 位置
                wrist_x, wrist_y, wrist_z = T_curr[:3, 3]
                pos_text = f"Pos: [{wrist_x:.3f}, {wrist_y:.3f}, {wrist_z:.3f}] m"
                cv2.putText(color_img, pos_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 2. 计算并显示手腕朝向（欧拉角）
                wrist_rot_matrix = T_curr[:3, :3]
                rotation = R.from_matrix(wrist_rot_matrix)
                euler_angles = rotation.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

                roll, pitch, yaw = euler_angles
                orient_text = f"Ori: [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]"
                cv2.putText(color_img, orient_text, (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 3. 绘制手腕坐标系方向指示器（小箭头）
                wrist_2d_x = int(keypoint_2d.landmark[0].x * w)
                wrist_2d_y = int(keypoint_2d.landmark[0].y * h)
                arrow_length = 40  # 像素

                # 提取旋转矩阵的列向量（局部坐标轴方向）
                x_axis = wrist_rot_matrix[:, 0]  # X 轴（红色）
                y_axis = wrist_rot_matrix[:, 1]  # Y 轴（绿色）
                z_axis = wrist_rot_matrix[:, 2]  # Z 轴（蓝色）

                # X 轴 - 红色箭头
                end_x = (
                    int(wrist_2d_x + x_axis[0] * arrow_length),
                    int(wrist_2d_y + x_axis[1] * arrow_length)
                )
                cv2.arrowedLine(color_img, (wrist_2d_x, wrist_2d_y), end_x,
                                (0, 0, 255), 2, tipLength=0.3)

                # Y 轴 - 绿色箭头
                end_y = (
                    int(wrist_2d_x + y_axis[0] * arrow_length),
                    int(wrist_2d_y + y_axis[1] * arrow_length)
                )
                cv2.arrowedLine(color_img, (wrist_2d_x, wrist_2d_y), end_y,
                                (0, 255, 0), 2, tipLength=0.3)

                # Z 轴 - 蓝色箭头
                end_z = (
                    int(wrist_2d_x + z_axis[0] * arrow_length),
                    int(wrist_2d_y + z_axis[1] * arrow_length)
                )
                cv2.arrowedLine(color_img, (wrist_2d_x, wrist_2d_y), end_z,
                                (255, 0, 0), 2, tipLength=0.3)

                cv2.putText(color_img, "X", (end_x[0] + 5, end_x[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(color_img, "Y", (end_y[0] + 5, end_y[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(color_img, "Z", (end_z[0] + 5, end_z[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if frame_count % 30 == 0:
                    logger.info(f"{ht} hand retargeting generated.")

            color_img = DualHandDetector.draw_skeleton_on_image(color_img, keypoint_2d_dict, style="default")


        else:
            cv2.putText(color_img, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for ht in retargeting_dict.keys():
            msg[ht] = last_state[ht]

        socket.send_json(msg)

        cv2.imshow("Teleop Server", color_img)
        if cv2.waitKey(1) == ord('q'):
            break


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: Literal["right", "left", "both"] = "both",
):
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    logger.info(f"启动视觉服务端...")
    logger.info(f"  机器人：{robot_name.value}")
    logger.info(f"  重定向类型：{retargeting_type.value}")
    logger.info(f"  手部模式：{hand_type}")

    config_paths = {}
    if hand_type in ["right", "both"]:
        config_paths["Right"] = str(get_default_config_path(robot_name, retargeting_type, HandType.right))
    if hand_type in ["left", "both"]:
        config_paths["Left"] = str(get_default_config_path(robot_name, retargeting_type, HandType.left))

    q = multiprocessing.Queue(maxsize=2)
    producer = multiprocessing.Process(target=produce_frames, args=(q,))
    consumer = multiprocessing.Process(target=start_vision_server, args=(q, str(robot_dir), config_paths))

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
