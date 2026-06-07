import numpy as np
from abc import ABC, abstractmethod

class BaseFormAnalyzer(ABC):
    def __init__(self):
        self.reset_rep_metrics()
        self.latest_summary = ["Waiting for activity..."]

    @abstractmethod
    def reset_rep_metrics(self):
        pass

    @abstractmethod
    def analyze_frame(self, landmarks, metrics, is_active):
        """
        Processes a single frame and returns real-time cues.
        """
        pass

    @abstractmethod
    def get_rep_summary(self):
        """
        Returns a summary after a rep is completed.
        """
        pass

class SquatFormAnalyzer(BaseFormAnalyzer):
    def __init__(self):
        super().__init__()

    def reset_rep_metrics(self):
        self.min_knee_angle = 180
        self.max_torso_tilt = 0
        self.max_heel_lift = 0
        self.rep_feedback = {
            "depth": "Target depth reached",
            "back": "Great upright posture",
            "feet": "Stable foot positioning",
        }

    def analyze_frame(self, landmarks, metrics, is_active):
        """
        landmarks: dict of relevant joint coords
        metrics: dict containing 'knee_angle'
        """
        if not is_active:
            return []

        knee_angle = metrics.get('knee_angle', 180)
        hip = landmarks.get('hip')
        shoulder = landmarks.get('shoulder')
        knee = landmarks.get('knee')
        ankle = landmarks.get('ankle')
        heel = landmarks.get('heel')

        if not (hip and shoulder and knee):
            return []

        # Track if we are descending or ascending
        # We are ascending if the current angle is significantly higher than the minimum reached this rep
        is_ascending = knee_angle > (self.min_knee_angle + 10)
        
        # 1. Track Depth
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle

        # 2. Analyze Back Posture
        dy = hip[1] - shoulder[1]
        dx = hip[0] - shoulder[0]
        
        # Torso tilt relative to vertical
        torso_tilt = np.abs(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))))
        
        if torso_tilt > self.max_torso_tilt:
            self.max_torso_tilt = torso_tilt

        # 3. Analyze Heel Lift (Distance-Invariant)
        heel_lifted = False
        if ankle and heel and knee:
            # Use lower leg length as a 'ruler' to normalize distance
            lower_leg_len = np.linalg.norm(np.array(knee) - np.array(ankle))
            
            if lower_leg_len > 10: # Avoid division by zero
                # Calculate vertical lift relative to ankle
                lift_amount = ankle[1] - heel[1]
                lift_ratio = lift_amount / lower_leg_len
                
                if lift_ratio > self.max_heel_lift:
                    self.max_heel_lift = lift_ratio
                
                # Threshold: If heel rises > 10% of leg length, it's a lift
                if lift_ratio > 0.10:
                    heel_lifted = True

        # 4. Real-time Cues
        cues = []
        if torso_tilt > 45: 
            cues.append("Keep chest up")
        
        if heel_lifted:
            cues.append("Keep heels down")
        
        # Real-time depth cues
        # PERSISTENT FEEDBACK LOGIC
        if not is_ascending:
            # We are in the descent or holding phase
            # Align 'Perfect depth' with the rep counter's target (95)
            if knee_angle <= 95: 
                cues.append("Perfect depth!")
            elif knee_angle < 105:
                cues.append("Almost there... go lower!")
            else:
                cues.append("Go deeper")
        else:
            # We are on the way back up
            # Only say "Good depth" if they actually hit the rep counter's target (95)
            if self.min_knee_angle <= 95:
                cues.append("Good depth! Drive up!")
            elif self.min_knee_angle < 110:
                cues.append("Shallow rep. Go lower next time.")
            else:
                cues.append("Finish the rep!")
        
        return cues

    def get_rep_summary(self):
        # Granular depth logic
        if self.min_knee_angle > 105:
            self.rep_feedback["depth"] = "Shallow depth - go lower"
        elif self.min_knee_angle > 90:
            self.rep_feedback["depth"] = "Depth: Acceptable"
        elif self.min_knee_angle > 70:
            self.rep_feedback["depth"] = "Depth: Excellent (Parallel)"
        else:
            self.rep_feedback["depth"] = "Depth: Deep Squat"
        
        # Back logic
        if self.max_torso_tilt > 50: # Relaxed summary threshold
            self.rep_feedback["back"] = "Excessive forward lean"
        elif self.max_torso_tilt > 35: # Relaxed summary threshold
            self.rep_feedback["back"] = "Chest could be higher"
        else:
            self.rep_feedback["back"] = "Great upright posture"

        # Heel logic (Using ratios now: 0.15 = 15% lift, 0.08 = 8% lift)
        if self.max_heel_lift > 0.15:
            self.rep_feedback["feet"] = "Heels lifted significantly"
        elif self.max_heel_lift > 0.08:
            self.rep_feedback["feet"] = "Slight heel instability"
        else:
            self.rep_feedback["feet"] = "Stable foot positioning"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_knee_angle, self.max_torso_tilt

