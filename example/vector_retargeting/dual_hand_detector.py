import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

# 右手 MANO 坐标系映射矩阵
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1], [-1, 0, 0],
        [0, 1, 0],
    ]
)

# 左手 MANO 坐标系映射矩阵
OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1], [1, 0, 0],
        [0, -1, 0],
    ]
)


class DualHandDetector:
    def __init__(
        self,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        # 开启双手检测 max_num_hands=2
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.inverse_hand_dict = {"Right": "Left", "Left": "Right"}

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoints_2d_dict, style="white"
    ):
        """
        在图像上绘制双手的骨架
        :param image: 原始图像
        :param keypoints_2d_dict: 包含双手 2D 关键点的字典 {"Left": keypoint_2d, "Right": keypoint_2d}
        """
        for hand_type, keypoint_2d in keypoints_2d_dict.items():
            # 不同的手可以设置不同的颜色，这里为了简单起见沿用统一的白色/默认风格
            if style == "default":
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            elif style == "white":
                landmark_style = {}
                for landmark in HandLandmark:
                    landmark_style[landmark] = DrawingSpec(
                        color=(255, 48, 48) if hand_type == "Right" else (48, 255, 48),  # 右手偏蓝/红，左手偏绿
                        circle_radius=4,
                        thickness=-1
                    )

                connections = hands_connections.HAND_CONNECTIONS
                connection_style = {}
                for pair in connections:
                    connection_style[pair] = DrawingSpec(thickness=2)

                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    keypoint_2d,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

        return image

    def detect(self, rgb):
        """
        检测双手
        :return: num_box, joint_pos_dict, keypoint_2d_dict, mediapipe_wrist_rot_dict
                 返回值均以字典形式按 "Left" 或 "Right" 分类。
        """
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, {}, {}, {}

        num_box = len(results.multi_hand_landmarks)

        joint_pos_dict = {}
        keypoint_2d_dict = {}
        wrist_rot_dict = {}

        for i in range(num_box):
            # 获取 MediaPipe 预测的左右手标签
            mp_label = results.multi_handedness[i].ListFields()[0][1][0].label

            # 修正标签：如果不是自拍模式，MediaPipe 的左右手判断需要反转
            actual_hand_type = mp_label if self.selfie else self.inverse_hand_dict[mp_label]

            keypoint_3d = results.multi_hand_world_landmarks[i]
            keypoint_2d = results.multi_hand_landmarks[i]

            # 解析 3D 关键点并以手腕为原点归一化
            keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

            # 估算手腕旋转矩阵
            mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

            # 根据当前是左手还是右手，选择不同的坐标系映射矩阵
            operator2mano = OPERATOR2MANO_RIGHT if actual_hand_type == "Right" else OPERATOR2MANO_LEFT
            joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano

            # 存入字典
            joint_pos_dict[actual_hand_type] = joint_pos
            keypoint_2d_dict[actual_hand_type] = keypoint_2d
            wrist_rot_dict[actual_hand_type] = mediapipe_wrist_rot

        return num_box, joint_pos_dict, keypoint_2d_dict, wrist_rot_dict

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        # 获取 手腕(0), 食指根部(5), 小拇指根部(9) 的点 （MediaPipe里 9 实际是中指根部，但原作者代码保留这个逻辑）
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
