import cv2
import numpy as np
from src.exercises.base import BaseExercise
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle
from src.analysis.rep_counter import TravelCounter
from src.analysis.form_analyzer import PushUpFormAnalyzer
from src.analysis.scoring_engine import PushUpScorer

class PushUp(BaseExercise):
    """
    Shoulder-Driven PRO Push-Up Engine.
    Powered by TravelCounter for robust vertical tracking.
    """
    def __init__(self):
        super().__init__()
        self.analyzer = PushUpFormAnalyzer()
        self.scorer = PushUpScorer()
        
        # PRO TravelCounter: 5% descent trigger, 4% return lockout, 0.15s min duration
        self.counter = TravelCounter(
            descent_thresh_pct=0.05,
            return_thresh_pct=0.04,
            min_duration=0.15
        )
        
        self.smoother = LandmarkSmoother(alpha=0.6, max_jump=100)
        
        # KEYPOINTS
        self.L_SHOULDER, self.R_SHOULDER = 11, 12
        self.L_ELBOW, self.R_ELBOW = 13, 14
        self.L_WRIST, self.R_WRIST = 15, 16
        self.L_HIP, self.R_HIP = 23, 24
        self.L_ANKLE, self.R_ANKLE = 27, 28
        
        self.KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28]
        self.last_valid_coords = {idx: None for idx in self.KEY_JOINTS}
        
        self.current_angle = 180
        self.working_side = "L"

    def process(self, landmarks, world_landmarks, w, h, timestamp_ms, img=None):
        self.warning_msg = None
        rep_done = False
        curr_time = timestamp_ms / 1000.0

        def get_pt(idx):
            lm = landmarks[idx]
            if lm.visibility >= 0.2:
                coords = (lm.x * w, lm.y * h)
                self.last_valid_coords[idx] = coords
                return coords
            return self.last_valid_coords[idx]

        # 1. Perspective Check (Lenient Sweet Spot)
        l_sh, r_sh = landmarks[self.L_SHOULDER], landmarks[self.R_SHOULDER]
        l_hip, r_hip = landmarks[self.L_HIP], landmarks[self.R_HIP]
        
        shoulder_width = abs(l_sh.x - r_sh.x) * w
        torso_len = np.linalg.norm(np.array([l_sh.x*w, l_sh.y*h]) - np.array([l_hip.x*w, l_hip.y*h]))
        
        if torso_len > 0 and (shoulder_width / torso_len > 0.75):
            self.warning_msg = "SIDE VIEW REQUIRED"
            return self.reps, self.state, False, ["Rotate slightly"], self.warning_msg, False

        # 2. Extract Key Points
        pts = {idx: self.smoother.smooth(idx, get_pt(idx)) for idx in self.KEY_JOINTS}
        
        if not (pts[self.L_SHOULDER] and pts[self.R_SHOULDER] and pts[self.L_ELBOW] and pts[self.R_ELBOW] and pts[self.L_HIP]):
            self.warning_msg = "INITIALIZING CORE JOINTS..."
            return self.reps, self.state, False, [], self.warning_msg, False

        # 3. Orientation Gate: Prone Position Required
        dx = abs(pts[self.L_SHOULDER][0] - pts[self.L_HIP][0])
        dy = abs(pts[self.L_SHOULDER][1] - pts[self.L_HIP][1])
        if dy > dx * 1.5: 
            self.warning_msg = "GET INTO PUSH-UP POSITION"
            return self.reps, self.state, False, ["Prone position required"], self.warning_msg, False

        # 4. Primary Signal: Shoulder Vertical Position
        shoulder_y = (pts[self.L_SHOULDER][1] + pts[self.R_SHOULDER][1]) / 2
        
        # 5. Secondary Signal: Elbow Flexion
        l_angle = compute_angle(pts[self.L_SHOULDER], pts[self.L_ELBOW], pts[self.L_WRIST]) if pts[self.L_WRIST] else 180
        r_angle = compute_angle(pts[self.R_SHOULDER], pts[self.R_ELBOW], pts[self.R_WRIST]) if pts[self.R_WRIST] else 180
        curr_min_elbow = min(l_angle, r_angle)
        
        if self.state == "UP":
            self.working_side = "L" if l_angle < r_angle else "R"

        # 6. Modular TravelCounter Update
        # We pass shoulder_y as the displacement signal
        self.reps, self.state, is_viable, rep_done, self.is_active, duration = self.counter.update(
            shoulder_y, torso_len, "HIGH", curr_time
        )

        # 7. Form Analysis & Dynamic Visualization
        feedback = []
        if is_viable:
            side = self.working_side
            SHOULDER = self.L_SHOULDER if side == "L" else self.R_SHOULDER
            ELBOW = self.L_ELBOW if side == "L" else self.R_ELBOW
            WRIST = self.L_WRIST if side == "L" else self.R_WRIST
            HIP = self.L_HIP if side == "L" else self.R_HIP
            ANKLE = self.L_ANKLE if side == "L" else self.R_ANKLE

            landmark_data = {
                'shoulder': pts[SHOULDER], 'elbow': pts[ELBOW], 
                'wrist': pts[WRIST], 'hip': pts[HIP], 'ankle': pts[ANKLE]
            }
            analysis_metrics = {'elbow_angle': curr_min_elbow, 'state': self.state}
            feedback = self.analyzer.analyze_frame(landmark_data, analysis_metrics, self.is_active)
            self.latest_feedback = feedback if self.is_active else self.latest_feedback

            if img is not None:
                # Add professional overlays (like angle text) if needed
                cv2.putText(img, f"{int(curr_min_elbow)}deg", (int(pts[ELBOW][0] + 5), int(pts[ELBOW][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if rep_done:
                summary, min_angle, max_sag = self.analyzer.get_rep_summary()
                self.latest_feedback = summary
                score_metrics = {'min_knee_angle': min_angle, 'max_torso_tilt': max_sag, 'duration': duration}
                self.last_score, _ = self.scorer.calculate_score(score_metrics)
                self.analyzer.reset_rep_metrics()

        return self.reps, self.state, self.is_active, self.latest_feedback, self.warning_msg, rep_done

    def get_metrics(self):
        return {
            "angle": self.min_elbow_angle if hasattr(self, 'min_elbow_angle') else 180,
            "side": self.working_side,
            "last_score": self.last_score,
            "last_rep_data": None
        }
