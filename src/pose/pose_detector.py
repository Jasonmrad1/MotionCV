import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.7, # Strict human recognition
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def find_pose(self, img, timestamp_ms):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        return result

    def draw_landmarks(self, img, result):
        if not result or not result.pose_landmarks:
            return img

        h, w, _ = img.shape
        # CORE SQUAT JOINTS ONLY (Shoulders, Hips, Knees, Ankles)
        KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]
        CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28)  # Right leg
        ]

        landmarks = result.pose_landmarks[0]

        # 1. Draw Skeleton (Subtle)
        for start_idx, end_idx in CONNECTIONS:
            if landmarks[start_idx].visibility > 0.2 and landmarks[end_idx].visibility > 0.2:
                pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(img, pt1, pt2, (150, 150, 150), 1)

        # 2. Draw Key Joints (Clean Green Dots)
        for idx in KEY_JOINTS:
            lm = landmarks[idx]
            if lm.visibility > 0.2:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        return img