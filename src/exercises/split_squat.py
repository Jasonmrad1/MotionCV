import cv2
import numpy as np
from src.exercises.base import BaseExercise
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle
from src.analysis.rep_counter import TravelCounter
from src.analysis.form_analyzer import LungeFormAnalyzer
from src.analysis.scoring_engine import LungeScorer

class SplitSquat(BaseExercise):
    """
    Hip-Driven Split Squat Engine.
    Powered by TravelCounter for robust vertical tracking.
    """
    def __init__(self):
        super().__init__()
        self.analyzer = LungeFormAnalyzer()
        self.scorer = LungeScorer()
        
        # PRO TravelCounter: 12% descent trigger, 5% return, 0.6s min duration
        self.counter = TravelCounter(
            descent_thresh_pct=0.12,
            return_thresh_pct=0.05,
            min_duration=0.6
        )
        
        self.smoother = LandmarkSmoother(alpha=0.3, max_jump=50)
        
        # KEYPOINTS
        self.L_SHOULDER, self.R_SHOULDER = 11, 12
        self.L_HIP, self.R_HIP = 23, 24
        self.L_KNEE, self.R_KNEE = 25, 26
        self.L_ANKLE, self.R_ANKLE = 27, 28
        self.L_HEEL, self.R_HEEL = 29, 30
        
        self.KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30]
        self.last_valid_coords = {idx: None for idx in self.KEY_JOINTS}
        
        self.min_knee_angle = 180
        self.last_working_side = "L"

    def process(self, landmarks, w, h, timestamp_ms, img=None):
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

        # 1. Perspective Check (Lenient Side View)
        l_sh, r_sh = landmarks[self.L_SHOULDER], landmarks[self.R_SHOULDER]
        l_hip, r_hip = landmarks[self.L_HIP], landmarks[self.R_HIP]
        torso_len = abs(l_sh.y - l_hip.y) * h
        shoulder_width = abs(l_sh.x - r_sh.x) * w
        
        if torso_len > 0 and (shoulder_width / torso_len > 0.75):
            self.warning_msg = "SIDE VIEW REQUIRED"
            return self.reps, self.state, False, ["Rotate slightly"], self.warning_msg, False

        # 2. Extract Key Points
        pts = {idx: self.smoother.smooth(idx, get_pt(idx)) for idx in self.KEY_JOINTS}
        
        if not (pts[self.L_HIP] and pts[self.R_HIP] and pts[self.L_KNEE] and pts[self.R_KNEE]):
            self.warning_msg = "INITIALIZING CORE JOINTS..."
            return self.reps, self.state, False, [], self.warning_msg, False

        # 3. Primary Signal: Hip Center Vertical Position
        hip_y = (pts[self.L_HIP][1] + pts[self.R_HIP][1]) / 2

        # 4. Secondary Signal: Knee Flexion
        l_angle = compute_angle(pts[self.L_HIP], pts[self.L_KNEE], pts[self.L_ANKLE]) if pts[self.L_ANKLE] else 180
        r_angle = compute_angle(pts[self.R_HIP], pts[self.R_KNEE], pts[self.R_ANKLE]) if pts[self.R_ANKLE] else 180
        curr_min_knee = min(l_angle, r_angle)
        
        if self.state == "UP":
            self.last_working_side = "L" if l_angle < r_angle else "R"

        # 5. TravelCounter Update
        self.reps, self.state, is_viable, rep_done, self.is_active, duration = self.counter.update(
            hip_y, torso_len, "HIGH", curr_time
        )

        # 6. Form Analysis & Feedback
        feedback = []
        if is_viable:
            side = self.last_working_side
            w_pts = {
                'hip': pts[self.L_HIP] if side == "L" else pts[self.R_HIP],
                'shoulder': pts[self.L_SHOULDER] if side == "L" else pts[self.R_SHOULDER],
                'knee': pts[self.L_KNEE] if side == "L" else pts[self.R_KNEE],
                'ankle': pts[self.L_ANKLE] if side == "L" else pts[self.R_ANKLE],
                'heel': pts[self.L_HEEL] if side == "L" else pts[self.R_HEEL]
            }
            analysis_metrics = {'knee_angle': curr_min_knee, 'state': self.state}
            feedback = self.analyzer.analyze_frame(w_pts, analysis_metrics, self.is_active)
            self.latest_feedback = feedback if self.is_active else self.latest_feedback

            if rep_done:
                summary, min_angle, max_tilt = self.analyzer.get_rep_summary()
                self.latest_feedback = summary
                score_metrics = {'min_knee_angle': min_angle, 'max_torso_tilt': max_tilt, 'duration': duration}
                self.last_score, _ = self.scorer.calculate_score(score_metrics)
                self.analyzer.reset_rep_metrics()

        return self.reps, self.state, self.is_active, self.latest_feedback, self.warning_msg, rep_done

    def get_metrics(self):
        return {
            "angle": self.min_knee_angle,
            "side": self.last_working_side,
            "last_score": self.last_score,
            "last_rep_data": None
        }
