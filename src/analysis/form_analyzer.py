import numpy as np

class FormAnalyzer:
    def __init__(self):
        self.reset_rep_metrics()
        self.latest_summary = ["Waiting for squat..."]

    def reset_rep_metrics(self):
        self.min_knee_angle = 180
        self.max_torso_tilt = 0
        self.rep_feedback = {
            "depth": "Target depth reached",
            "back": "Neutral spine maintained",
        }

    def analyze_posture(self, hip, shoulder, knee, ankle, knee_angle, is_active):
        """
        Collects posture data only during an active squat.
        Returns real-time cues if active.
        """
        if not is_active:
            return []

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

        # 3. Real-time Cues
        cues = []
        if torso_tilt > 30: 
            cues.append("Keep chest up")
        
        # Real-time depth cues
        # Image coordinates: Y increases downwards. hip[1] > knee[1] means hip is lower than knee.
        hip_to_knee_rel = hip[1] - knee[1]
        
        if knee_angle > 110:
            cues.append("Go deeper")
        elif hip_to_knee_rel > 0: # Hip below knee
            cues.append("Excellent depth!")
        elif knee_angle < 95:
            cues.append("Target depth reached")
        
        return cues

    def get_rep_summary(self):
        """
        Produces a final summary after a validated rep.
        Returns: (feedback_list, min_knee_angle, max_torso_tilt)
        """
        # More granular depth logic
        if self.min_knee_angle > 105:
            self.rep_feedback["depth"] = "Shallow depth - go lower"
        elif self.min_knee_angle > 90:
            self.rep_feedback["depth"] = "Depth: Acceptable"
        elif self.min_knee_angle > 70:
            self.rep_feedback["depth"] = "Depth: Excellent (Parallel)"
        else:
            self.rep_feedback["depth"] = "Depth: Deep Squat"
        
        # Back logic
        if self.max_torso_tilt > 40:
            self.rep_feedback["back"] = "Excessive forward lean"
        elif self.max_torso_tilt > 25:
            self.rep_feedback["back"] = "Chest could be higher"
        else:
            self.rep_feedback["back"] = "Great upright posture"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_knee_angle, self.max_torso_tilt
