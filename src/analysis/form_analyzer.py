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
        if not is_active:
            return []

        knee_angle = metrics.get('knee_angle', 180)
        state = metrics.get('state', 'UP')
        hip = landmarks.get('hip')
        shoulder = landmarks.get('shoulder')
        knee = landmarks.get('knee')
        ankle = landmarks.get('ankle')
        heel = landmarks.get('heel')

        if hip is None or shoulder is None or knee is None:
            return []

        is_ascending = (knee_angle > self.min_knee_angle + 3) or (state == 'UP')
        
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle

        dy = hip[1] - shoulder[1]
        dx = hip[0] - shoulder[0]
        torso_tilt = np.abs(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))))
        
        if torso_tilt > self.max_torso_tilt:
            self.max_torso_tilt = torso_tilt

        heel_lifted = False
        if ankle and heel and knee:
            lower_leg_len = np.linalg.norm(np.array(knee) - np.array(ankle))
            if lower_leg_len > 10:
                lift_amount = ankle[1] - heel[1]
                lift_ratio = lift_amount / lower_leg_len
                if lift_ratio > self.max_heel_lift:
                    self.max_heel_lift = lift_ratio
                if lift_ratio > 0.10:
                    heel_lifted = True

        cues = []
        if torso_tilt > 45: cues.append("Keep chest up")
        if heel_lifted: cues.append("Keep heels down")
        
        if not is_ascending:
            if knee_angle <= 95: cues.append("Perfect depth!")
            elif knee_angle < 105: cues.append("Almost there... go lower!")
            else: cues.append("Go deeper")
        else:
            if self.min_knee_angle <= 95: cues.append("Good depth! Drive up!")
            elif self.min_knee_angle < 110: cues.append("Shallow rep. Go lower next time.")
            else: cues.append("Finish the rep!")
        
        return cues

    def get_rep_summary(self):
        if self.min_knee_angle > 105: self.rep_feedback["depth"] = "Shallow depth - go lower"
        elif self.min_knee_angle > 90: self.rep_feedback["depth"] = "Depth: Acceptable"
        elif self.min_knee_angle > 70: self.rep_feedback["depth"] = "Depth: Excellent (Parallel)"
        else: self.rep_feedback["depth"] = "Depth: Deep Squat"
        
        if self.max_torso_tilt > 50: self.rep_feedback["back"] = "Excessive forward lean"
        elif self.max_torso_tilt > 35: self.rep_feedback["back"] = "Chest could be higher"
        else: self.rep_feedback["back"] = "Great upright posture"

        if self.max_heel_lift > 0.15: self.rep_feedback["feet"] = "Heels lifted significantly"
        elif self.max_heel_lift > 0.08: self.rep_feedback["feet"] = "Slight heel instability"
        else: self.rep_feedback["feet"] = "Stable foot positioning"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_knee_angle, self.max_torso_tilt

class LungeFormAnalyzer(BaseFormAnalyzer):
    def __init__(self):
        super().__init__()

    def reset_rep_metrics(self):
        self.min_knee_angle = 180
        self.max_torso_tilt = 0
        self.max_heel_lift = 0
        self.rep_feedback = {
            "depth": "Target depth reached",
            "back": "Great upright posture",
            "stability": "Stable foot positioning",
        }

    def analyze_frame(self, landmarks, metrics, is_active):
        if not is_active:
            return []

        knee_angle = metrics.get('knee_angle', 180)
        state = metrics.get('state', 'UP')
        hip = landmarks.get('hip')
        shoulder = landmarks.get('shoulder')
        knee = landmarks.get('knee')
        ankle = landmarks.get('ankle')
        heel = landmarks.get('heel')

        if not (hip and knee):
            return []

        is_ascending = (knee_angle > self.min_knee_angle + 3) or (state == 'UP')

        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle

        torso_tilt = 0
        if shoulder:
            dy, dx = hip[1] - shoulder[1], hip[0] - shoulder[0]
            torso_tilt = np.abs(np.degrees(np.arctan2(np.abs(dx), np.abs(dy))))
            if torso_tilt > self.max_torso_tilt:
                self.max_torso_tilt = torso_tilt

        heel_lifted = False
        if ankle and heel and knee:
            lower_leg_len = np.linalg.norm(np.array(knee) - np.array(ankle))
            if lower_leg_len > 10:
                lift_amount = ankle[1] - heel[1]
                lift_ratio = lift_amount / lower_leg_len
                if lift_ratio > self.max_heel_lift:
                    self.max_heel_lift = lift_ratio
                if lift_ratio > 0.15:
                    heel_lifted = True

        cues = []
        if torso_tilt > 35: cues.append("Keep chest up")
        if heel_lifted: cues.append("Front heel down!")
        
        if not is_ascending:
            if knee_angle > 125: cues.append("Go deeper")
            elif knee_angle > 118: cues.append("Almost there...")
            else: cues.append("Perfect depth!")
        else:
            if self.min_knee_angle <= 118: cues.append("Good depth! Drive up!")
            else: cues.append("Finish the rep!")
            
        return cues

    def get_rep_summary(self):
        if self.min_knee_angle > 125: self.rep_feedback["depth"] = "Shallow depth - go lower"
        elif self.min_knee_angle > 110: self.rep_feedback["depth"] = "Excellent depth"
        else: self.rep_feedback["depth"] = "Deep movement"
        
        if self.max_torso_tilt > 40: self.rep_feedback["back"] = "Leaning too far forward"
        else: self.rep_feedback["back"] = "Great upright posture"

        if self.max_heel_lift > 0.12: self.rep_feedback["stability"] = "Front heel lifted"
        else: self.rep_feedback["stability"] = "Stable front foot"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_knee_angle, self.max_torso_tilt

class PushUpFormAnalyzer(BaseFormAnalyzer):
    def __init__(self):
        super().__init__()
        self.cue_streaks = {}
        self.active_cue = None
        self.persistence_thresh = 6 
        self.last_angle = 180

    def reset_rep_metrics(self):
        self.min_elbow_angle = 180
        self.max_back_sag = 0
        self.max_elbow_flare = 0
        self.rep_feedback = {
            "depth": "Great depth",
            "back": "Strong plank position",
            "elbows": "Perfect elbow tuck",
        }
        self.cue_streaks = {}
        self.active_cue = None

    def analyze_frame(self, landmarks, metrics, is_active):
        if not is_active:
            self.active_cue = None
            return []

        elbow_angle = metrics.get('elbow_angle', 180)
        state = metrics.get('state', 'UP')
        shoulder = landmarks.get('shoulder')
        elbow = landmarks.get('elbow')
        wrist = landmarks.get('wrist')
        hip = landmarks.get('hip')
        ankle = landmarks.get('ankle')

        if not (shoulder and elbow and hip and ankle):
            return []

        is_ascending = (elbow_angle > self.min_elbow_angle + 3) or (state == 'UP')
        angular_velocity = self.last_angle - elbow_angle 
        self.last_angle = elbow_angle

        if elbow_angle < self.min_elbow_angle:
            self.min_elbow_angle = elbow_angle

        from src.analysis.angle_utils import compute_angle
        body_alignment = compute_angle(shoulder, hip, ankle)
        alignment_error = abs(180 - body_alignment)
        if alignment_error > self.max_back_sag:
            self.max_back_sag = alignment_error

        flare_angle = compute_angle(hip, shoulder, elbow)
        if flare_angle > self.max_elbow_flare:
            self.max_elbow_flare = flare_angle

        candidate_cues = []
        if flare_angle > 75: candidate_cues.append("Tuck elbows in")
        elif alignment_error > 25: candidate_cues.append("Core tight - hips up")
        elif not is_ascending:
            if angular_velocity > 6: candidate_cues.append("Slow down (Control)")
            elif elbow_angle > 115: candidate_cues.append("Go lower")
            elif elbow_angle > 85: candidate_cues.append("Almost there...")
            else: candidate_cues.append("Perfect depth!")
        else:
            if self.min_elbow_angle <= 90: candidate_cues.append("Explosive up!")
            else: candidate_cues.append("Finish the rep!")

        if not candidate_cues:
            self.active_cue = None
            return []

        primary_candidate = candidate_cues[0]
        self.cue_streaks[primary_candidate] = self.cue_streaks.get(primary_candidate, 0) + 1
        
        if len(self.cue_streaks) > 1:
            for cue in list(self.cue_streaks.keys()):
                if cue != primary_candidate: self.cue_streaks[cue] = 0

        if self.cue_streaks[primary_candidate] >= self.persistence_thresh:
            self.active_cue = primary_candidate
        
        return [self.active_cue] if self.active_cue else []

    def get_rep_summary(self):
        if self.min_elbow_angle > 105: self.rep_feedback["depth"] = "Go lower for full range"
        else: self.rep_feedback["depth"] = "Excellent range of motion"

        if self.max_back_sag > 20: self.rep_feedback["back"] = "Avoid sagging/piking"
        else: self.rep_feedback["back"] = "Great plank alignment"

        if self.max_elbow_flare > 70: self.rep_feedback["elbows"] = "Elbows flared too wide"
        else: self.rep_feedback["elbows"] = "Safe elbow placement"

        self.latest_summary = list(self.rep_feedback.values())
        return self.latest_summary, self.min_elbow_angle, self.max_back_sag

class CurlFormAnalyzer(BaseFormAnalyzer):
    def __init__(self):
        super().__init__()
        self.cue_streaks = {}
        self.active_cue = None
        self.persistence_thresh = 5
        self.decay_rate = 0.5 
        self.last_angle = 180
        self.last_time = 0
        self.last_side = None

    def reset_rep_metrics(self):
        self.min_elbow_angle = 180
        self.max_elbow_drift = 0 
        self.max_torso_tilt = 0  
        self.max_shoulder_tilt = 0 
        self.rep_feedback = {
            "rom": "Full range of motion",
            "elbows": "Solid elbow stability",
            "posture": "Strong upright posture",
        }
        self.cue_streaks = {}
        self.active_cue = None
        self.last_angle = 180

    def analyze_frame(self, landmarks, metrics, is_active):
        if not is_active:
            self.active_cue = None
            return []

        import time
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        
        elbow_angle = metrics.get('elbow_angle', 180)
        state = metrics.get('state', 'NEUTRAL')
        side = metrics.get('side', 'L')
        
        shoulder = metrics.get('shoulder')
        elbow = metrics.get('elbow')
        hip = metrics.get('hip')
        l_sh = metrics.get('l_shoulder')
        r_sh = metrics.get('r_shoulder')

        if shoulder is None or elbow is None or hip is None:
            return []

        if side != self.last_side:
            self.last_angle = elbow_angle
            self.last_side = side

        # 1. Biometrics
        angular_velocity = (self.last_angle - elbow_angle) / dt 
        self.last_angle = elbow_angle
        self.last_time = now

        # Torso Tilt
        dy, dx = hip[1] - shoulder[1], hip[0] - shoulder[0]
        torso_tilt = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
        if torso_tilt > self.max_torso_tilt: self.max_torso_tilt = torso_tilt

        # Elbow Drift (Stability)
        drift = abs(shoulder[0] - elbow[0])
        if drift > self.max_elbow_drift: self.max_elbow_drift = drift

        if elbow_angle < self.min_elbow_angle: self.min_elbow_angle = elbow_angle

        # 2. Coaching Logic (PRIORITIZED)
        candidate_cues = []
        
        # Priority 1: Momentum/Cheating (SAFETY)
        if torso_tilt > 12:
            candidate_cues.append("Keep torso still - no momentum")
        elif angular_velocity > 130 and elbow_angle > 60:
            candidate_cues.append("Control the weight - slow down")
        
        # Priority 2: Alignment
        elif l_sh is not None and r_sh is not None and abs(l_sh[1] - r_sh[1]) > 0.05:
            candidate_cues.append("Level your shoulders")
        
        # Priority 3: Technical Form
        elif drift > 0.15: 
            candidate_cues.append("Keep elbows pinned to ribs")
        elif state == "ASCENDING" and elbow_angle > 90:
            candidate_cues.append("Squeeze all the way up")
        elif state == "DESCENDING" and elbow_angle < 130:
            candidate_cues.append("Full extension at the bottom")

        if not candidate_cues:
            self.active_cue = None
            return []

        # 3. Persistence Filter
        primary = candidate_cues[0]
        self.cue_streaks[primary] = self.cue_streaks.get(primary, 0) + 1
        
        # Decay other cues
        for c in list(self.cue_streaks.keys()):
            if c != primary:
                self.cue_streaks[c] = max(0, self.cue_streaks[c] - self.decay_rate)

        if self.cue_streaks[primary] >= self.persistence_thresh:
            self.active_cue = primary
        
        return [self.active_cue] if self.active_cue else []

    def get_rep_summary(self):
        # ROM Alignment with Scorer (45 perfect, 110 fail)
        if self.min_elbow_angle > 90: rom = "Incomplete"
        elif self.min_elbow_angle > 70: rom = "Partial"
        elif self.min_elbow_angle > 45: rom = "Good"
        else: rom = "Excellent (Full)"

        # Stability Alignment (0.12 perfect, 0.30 fail)
        if self.max_elbow_drift > 0.25: stability = "Significant Drift"
        elif self.max_elbow_drift > 0.18: stability = "Minor Drift"
        elif self.max_elbow_drift > 0.12: stability = "Good"
        else: stability = "Solid"
        
        # Posture Alignment (10 perfect, 25 fail)
        if self.max_torso_tilt > 20: posture = "Heavy Momentum"
        elif self.max_torso_tilt > 12: posture = "Slight Swing"
        elif self.max_torso_tilt > 8: posture = "Good"
        else: posture = "Solid"

        return [f"ROM: {rom}", f"Stability: {stability}", f"Posture: {posture}"], self.min_elbow_angle, self.max_elbow_drift, self.max_torso_tilt
