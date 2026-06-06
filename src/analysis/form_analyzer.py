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

    def analyze_posture(self, hip, shoulder, knee_angle, is_active):
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
        # Using a more precise vertical reference
        dy = hip[1] - shoulder[1]
        dx = hip[0] - shoulder[0]
        
        # Torso tilt relative to vertical (ideal is close to 0-15 degrees)
        torso_tilt = np.abs(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))))
        
        if torso_tilt > self.max_torso_tilt:
            self.max_torso_tilt = torso_tilt

        # 3. Real-time Cues
        cues = []
        if torso_tilt > 25: # More precise threshold
            cues.append("Keep chest up")
        
        # Real-time depth cues (more responsive)
        if knee_angle > 105:
            cues.append("Go deeper")
        elif knee_angle < 90:
            cues.append("Good depth!")
        
        return cues

    def get_rep_summary(self):
        """
        Produces a final summary after a validated rep.
        Returns: (feedback_list, min_knee_angle, max_torso_tilt)
        """
        # Depth logic
        if self.min_knee_angle > 100:
            self.rep_feedback["depth"] = "Go deeper next time"
        elif self.min_knee_angle < 75:
            self.rep_feedback["depth"] = "Excellent depth!"
        else:
            self.rep_feedback["depth"] = "Target depth reached"
        
        # Back logic
        if self.max_torso_tilt > 45:
            self.rep_feedback["back"] = "Avoid excessive leaning"
        elif self.max_torso_tilt > 30:
            self.rep_feedback["back"] = "Maintain upright chest"
        else:
            self.rep_feedback["back"] = "Neutral spine maintained"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_knee_angle, self.max_torso_tilt
