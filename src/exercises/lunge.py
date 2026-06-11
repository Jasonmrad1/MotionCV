import cv2
import numpy as np
from src.exercises.base import BaseExercise
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle
from src.analysis.rep_counter import AngleCounter
from src.analysis.form_analyzer import LungeFormAnalyzer
from src.analysis.scoring_engine import LungeScorer

class Lunge(BaseExercise):
    """
    Modular Lunge Engine for Forward, Reverse, and Walking Lunges.
    Optimized for dynamic stepping movements.
    """
    def __init__(self):
        super().__init__()
        # Standard Lunge Thresholds
        self.counter = AngleCounter(
            down_thresh=120,   # Standard descent trigger
            up_thresh=155,     # Full lockout preferred for dynamic lunges
            target_depth=110,  # Standard depth target
            min_rom=35
        )
        self.analyzer = LungeFormAnalyzer()
        self.scorer = LungeScorer()
        # Responsive smoothing (0.5 alpha) for dynamic steps
        self.smoother = LandmarkSmoother(alpha=0.5, max_jump=80)
        
        # KEYPOINT MAP
        self.L_SHOULDER, self.R_SHOULDER = 11, 12
        self.L_HIP, self.R_HIP = 23, 24
        self.L_KNEE, self.R_KNEE = 25, 26
        self.L_ANKLE, self.R_ANKLE = 27, 28
        self.L_HEEL, self.R_HEEL = 29, 30
        
        self.L_INDICES = [self.L_HIP, self.L_KNEE, self.L_ANKLE]
        self.R_INDICES = [self.R_HIP, self.R_KNEE, self.R_ANKLE]
        self.KEY_JOINTS = [
            self.L_SHOULDER, self.R_SHOULDER, 
            self.L_HIP, self.R_HIP, 
            self.L_KNEE, self.R_KNEE, 
            self.L_ANKLE, self.R_ANKLE, 
            self.L_HEEL, self.R_HEEL
        ]
        self.last_valid_coords = {idx: None for idx in self.KEY_JOINTS}
        self.current_angle = 180
        self.side_prefix = None

    def process(self, landmarks, world_landmarks, w, h, timestamp_ms, img=None):
        self.warning_msg = None
        rep_done = False
        
        # 1. Perspective Check (Lenient Side View)
        l_sh, r_sh = landmarks[self.L_SHOULDER], landmarks[self.R_SHOULDER]
        l_hip, r_hip = landmarks[self.L_HIP], landmarks[self.R_HIP]
        torso_len = abs(l_sh.y - l_hip.y) * h
        shoulder_width = abs(l_sh.x - r_sh.x) * w
        
        # Relaxed 0.75 threshold for better user experience
        if torso_len > 0 and (shoulder_width / torso_len > 0.75):
            self.warning_msg = "SIDE VIEW REQUIRED"
            return self.reps, self.state, False, ["Rotate slightly"], self.warning_msg, False

        # 2. Dynamic Side Selection with Locking
        if not self.is_active and self.state == "UP":
            self.side_prefix, conf_level = self.get_best_side(landmarks, self.L_INDICES, self.R_INDICES)
        else:
            _, conf_level = self.get_best_side(landmarks, self.L_INDICES, self.R_INDICES)
        
        # Select joints based on selected side
        HIP = self.L_HIP if self.side_prefix == "L" else self.R_HIP
        KNEE = self.L_KNEE if self.side_prefix == "L" else self.R_KNEE
        ANKLE = self.L_ANKLE if self.side_prefix == "L" else self.R_ANKLE
        SHOULDER = self.L_SHOULDER if self.side_prefix == "L" else self.R_SHOULDER
        HEEL = self.L_HEEL if self.side_prefix == "L" else self.R_HEEL

        def get_target_pt(idx, is_leg=True):
            current_v = landmarks[idx].visibility
            if current_v >= 0.4:
                coords = (landmarks[idx].x * w, landmarks[idx].y * h)
                self.last_valid_coords[idx] = coords
                return coords
            if is_leg and current_v < 0.1: return None
            return self.last_valid_coords[idx]

        # 3. Extract and Smooth
        raw_hip = get_target_pt(HIP)
        raw_knee = get_target_pt(KNEE)
        raw_ankle = get_target_pt(ANKLE)
        raw_heel = get_target_pt(HEEL)
        shoulder = get_target_pt(SHOULDER, is_leg=False)

        hip = self.smoother.smooth(HIP, raw_hip)
        knee = self.smoother.smooth(KNEE, raw_knee)
        ankle = self.smoother.smooth(ANKLE, raw_ankle)
        heel = self.smoother.smooth(HEEL, raw_heel)

        if hip and knee and ankle and conf_level != "LOW":
            self.current_angle = compute_angle(hip, knee, ankle)
            
            # 4. Rep Counter Update
            self.reps, self.state, is_viable, rep_done, self.is_active, duration = self.counter.update(
                self.current_angle, conf_level, timestamp_ms / 1000.0
            )

            if is_viable:
                # 5. Form Analysis
                landmark_data = {
                    'hip': hip, 'shoulder': shoulder, 
                    'knee': knee, 'ankle': ankle, 'heel': heel
                }
                analysis_metrics = {
                    'knee_angle': self.current_angle,
                    'state': self.state
                }
                realtime_cues = self.analyzer.analyze_frame(landmark_data, analysis_metrics, self.is_active)
                self.latest_feedback = realtime_cues if self.is_active else self.latest_feedback

                if rep_done:
                    summary_feedback, min_angle, max_tilt = self.analyzer.get_rep_summary()
                    self.latest_feedback = summary_feedback
                    score_metrics = {'min_knee_angle': min_angle, 'max_torso_tilt': max_tilt, 'duration': duration}
                    self.last_score, _ = self.scorer.calculate_score(score_metrics)
                    self.analyzer.reset_rep_metrics()

                if conf_level == "MEDIUM":
                    self.warning_msg = f"REDUCED VISIBILITY ({self.side_prefix})"
            else:
                self.warning_msg = "POSE LOST"
        else:
            self.warning_msg = "INITIALIZING CORE JOINTS..."
            
        return self.reps, self.state, self.is_active, self.latest_feedback, self.warning_msg, rep_done

    def get_metrics(self):
        return {
            "angle": self.current_angle,
            "side": self.side_prefix,
            "last_score": self.last_score,
            "last_rep_data": self.counter.last_rep_data
        }
