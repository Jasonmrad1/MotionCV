import cv2
import numpy as np
import time
from src.exercises.base import BaseExercise
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle_2d
from src.analysis.form_analyzer import CurlFormAnalyzer
from src.analysis.scoring_engine import CurlScorer

class CurlState:
    NEUTRAL = "NEUTRAL"
    ASCENDING = "ASCENDING"
    PEAK = "PEAK"
    DESCENDING = "DESCENDING"

class RobustCurlCounter:
    """
    Independent arm tracking with fast reactivity and K-frame hysteresis.
    """
    def __init__(self, side="L", k_frames=4):
        self.side = side
        self.state = CurlState.NEUTRAL
        self.pending_state = CurlState.NEUTRAL
        self.state_frames = 0
        self.k_frames = k_frames
        
        self.count = 0
        self.start_time = None
        self.min_angle = 180
        self.is_active = False
        self.last_rep_data = None # Storage for scoring metrics
        
        # Thresholds
        self.START_THRESH = 145
        self.FLEX_THRESH = 40
        self.EXTEND_THRESH = 155

    def update(self, angle, timestamp):
        rep_completed = False
        target_state = self.state
        
        if self.state == CurlState.NEUTRAL:
            if angle < self.START_THRESH:
                target_state = CurlState.ASCENDING
        elif self.state == CurlState.ASCENDING:
            if angle < self.FLEX_THRESH:
                target_state = CurlState.PEAK
            elif angle > self.EXTEND_THRESH + 5:
                target_state = CurlState.NEUTRAL
        elif self.state == CurlState.PEAK:
            if angle > (self.FLEX_THRESH + 20):
                target_state = CurlState.DESCENDING
        elif self.state == CurlState.DESCENDING:
            if angle > self.EXTEND_THRESH:
                target_state = CurlState.NEUTRAL

        # Apply K-Frame Hysteresis
        if target_state != self.state:
            if self.pending_state != target_state:
                self.pending_state = target_state
                self.state_frames = 1
            else:
                self.state_frames += 1
                if self.state_frames >= self.k_frames:
                    prev_state = self.state
                    self.state = target_state
                    
                    # Transition Side Effects
                    if self.state == CurlState.ASCENDING and prev_state == CurlState.NEUTRAL:
                        self.start_time = timestamp
                        self.min_angle = angle
                        self.is_active = True
                    elif self.state == CurlState.NEUTRAL and prev_state == CurlState.DESCENDING:
                        duration = timestamp - self.start_time
                        if duration > 0.4:
                            self.count += 1
                            rep_completed = True
                            self.last_rep_data = {
                                "min_angle": self.min_angle,
                                "duration": duration
                            }
                        self.is_active = False
                        self.min_angle = 180
                    elif self.state == CurlState.NEUTRAL and prev_state == CurlState.ASCENDING:
                        self.is_active = False
                        self.min_angle = 180
        else:
            self.pending_state = self.state
            self.state_frames = 0
            
        # Continuous updates (regardless of state transition delay)
        if self.state == CurlState.ASCENDING or self.pending_state == CurlState.ASCENDING:
            self.min_angle = min(self.min_angle, angle)

        return self.count, self.state, rep_completed, self.is_active

class Curl(BaseExercise):
    """
    Reliable Bicep Curl Engine.
    - Zero-lag angle tracking with One Euro Filter.
    - Independent L/R counting.
    - Side-view skeleton cleanup.
    """
    def __init__(self):
        super().__init__()
        self.l_counter = RobustCurlCounter(side="L")
        self.r_counter = RobustCurlCounter(side="R")
        
        self.analyzer = CurlFormAnalyzer()
        self.scorer = CurlScorer()
        
        # Reactive beta (0.1) for zero lag
        self.smoother = LandmarkSmoother(min_cutoff=0.8, beta=0.1)

        # Indices
        self.L_S, self.R_S = 11, 12
        self.L_E, self.R_E = 13, 14
        self.L_W, self.R_W = 15, 16
        self.L_H, self.R_H = 23, 24
        
        self.BASE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
        self.KEY_JOINTS = list(self.BASE_JOINTS)
        
        self.l_angle = 180
        self.r_angle = 180
        self.active_side = "L"
        self.view_mode = "FRONT"
        self.last_global_rep_time = 0

    def process(self, landmarks, world_landmarks, w, h, timestamp_ms, img=None):
        curr_time = timestamp_ms / 1000.0
        self.smoother.start_frame()
        
        # 1. Perspective Selection
        l_sh, r_sh = landmarks[self.L_S], landmarks[self.R_S]
        l_hip, r_hip = landmarks[self.L_H], landmarks[self.R_H]
        torso_len = max(np.sqrt((l_sh.x - l_hip.x)**2 + (l_sh.y - l_hip.y)**2), 0.01)
        shoulder_w = abs(l_sh.x - r_sh.x)
        self.view_mode = "FRONT" if (shoulder_w / torso_len > 0.35) else "SIDE"

        # 2. Extract & Smooth Points
        pts = {}
        for idx in self.BASE_JOINTS:
            lm = landmarks[idx]
            if lm.visibility > 0.15:
                pts[idx] = self.smoother.smooth(idx, (lm.x, lm.y, lm.z))
            else:
                pts[idx] = self.smoother.smoothed_landmarks.get(idx, (lm.x, lm.y, lm.z))

        # 3. Final Angle Calculation
        self.l_angle = compute_angle_2d(pts[self.L_S][:2], pts[self.L_E][:2], pts[self.L_W][:2])
        self.r_angle = compute_angle_2d(pts[self.R_S][:2], pts[self.R_E][:2], pts[self.R_W][:2])

        # 4. Updates
        l_c, l_st, l_dn, l_act = self.l_counter.update(self.l_angle, curr_time)
        r_c, r_st, r_dn, r_act = self.r_counter.update(self.r_angle, curr_time)

        # Sync (0.4s window)
        rep_done = False
        if l_dn or r_dn:
            if curr_time - self.last_global_rep_time > 0.4:
                self.reps += 1
                self.last_global_rep_time = curr_time
                rep_done = True

        self.is_active = l_act or r_act

        # 5. Side Selection & UI Logic (Simplified)
        if not self.is_active:
            if self.view_mode == "SIDE":
                # Pick the arm closer to the camera (higher visibility)
                self.active_side = "L" if landmarks[self.L_S].visibility > landmarks[self.R_S].visibility else "R"
            else:
                # FRONT view: Pick side that is flexing more
                self.active_side = "L" if self.l_angle < self.r_angle else "R"

        if self.view_mode == "SIDE":
            self.KEY_JOINTS = [11, 12, 23, 24]
            arm = [13, 15] if self.active_side == "L" else [14, 16]
            self.KEY_JOINTS.extend(arm)
        else:
            self.KEY_JOINTS = list(self.BASE_JOINTS)

        self.state = l_st if self.active_side == "L" else r_st
        
        # 6. Feedback & Form Analysis
        side = self.active_side
        side_pts = (self.L_S, self.L_E, self.L_W, self.L_H) if side == "L" else (self.R_S, self.R_E, self.R_W, self.R_H)
        shoulder, elbow, wrist, hip = [pts[idx] for idx in side_pts]
        
        metrics = {
            'elbow_angle': self.l_angle if side == "L" else self.r_angle,
            'state': self.state,
            'view_mode': self.view_mode,
            'side': side,
            'shoulder': shoulder, 'elbow': elbow, 'wrist': wrist, 'hip': hip,
            'l_shoulder': pts[self.L_S], 'r_shoulder': pts[self.R_S]
        }
        
        feedback = self.analyzer.analyze_frame(pts, metrics, self.is_active)
        if self.is_active:
            self.latest_feedback = feedback
        else:
            if not rep_done:
                self.latest_feedback = []

        if img is not None:
            self._draw_debug(img, pts, self.active_side, w, h)

        if rep_done:
            summary, min_angle, max_drift, max_tilt = self.analyzer.get_rep_summary()
            counter_obj = self.l_counter if l_dn else self.r_counter
            duration = counter_obj.last_rep_data["duration"] if counter_obj.last_rep_data else 1.0
            
            score_metrics = {
                'min_elbow_angle': min_angle, 
                'max_drift': max_drift,
                'max_tilt': max_tilt, 
                'duration': duration
            }
            self.last_score, breakdown = self.scorer.calculate_score(score_metrics)
            
            self.latest_feedback = summary
            if breakdown.get("Penalty", 0) > 0:
                if duration < 1.0:
                    self.latest_feedback.append("Penalty: Rep too fast (momentum)")
            
            if self.last_score < 80:
                if breakdown["ROM"] < 30: self.latest_feedback.append("Tip: Squeeze higher at the top")
                if breakdown["Stability"] < 20: self.latest_feedback.append("Tip: Keep elbows pinned to ribs")
                if breakdown["Posture"] < 20: self.latest_feedback.append("Tip: Don't swing your torso")

            self.analyzer.reset_rep_metrics()

        return self.reps, self.state, self.is_active, self.latest_feedback, None, rep_done

    def _draw_debug(self, img, pts, active_side, w, h):
        for side in ["L", "R"]:
            E_idx = self.L_E if side == "L" else self.R_E
            angle = self.l_angle if side == "L" else self.r_angle
            color = (0, 255, 0) if (side == active_side and self.is_active) else (255, 255, 255)
            pos = (int(pts[E_idx][0] * w), int(pts[E_idx][1] * h))
            cv2.putText(img, f"{int(angle)}", (pos[0] + 10, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def get_metrics(self):
        return {
            "angle": self.l_angle if self.active_side == "L" else self.r_angle,
            "side": self.active_side,
            "last_score": self.last_score
        }
