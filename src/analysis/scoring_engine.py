from abc import ABC, abstractmethod

class BaseScorer(ABC):
    @abstractmethod
    def calculate_score(self, metrics):
        """
        Calculates a score (0-100) based on exercise-specific metrics.
        Returns: (total_score, breakdown_dict)
        """
        pass

class SquatScorer(BaseScorer):
    def calculate_score(self, metrics):
        min_knee_angle = metrics.get('min_knee_angle', 180)
        max_torso_tilt = metrics.get('max_torso_tilt', 0)
        rep_duration = metrics.get('duration', 0)

        # 1. Depth Score (30 points)
        depth_score = 0
        if min_knee_angle < 80: depth_score = 30
        elif min_knee_angle > 110: depth_score = 0
        else: depth_score = 30 * (1 - (min_knee_angle - 80) / 30)

        # 2. Posture Score (40 points)
        posture_score = 0
        if max_torso_tilt < 30: posture_score = 40
        elif max_torso_tilt > 60: posture_score = 0
        else: posture_score = 40 * (1 - (max_torso_tilt - 30) / 30)

        # 3. Tempo Score (30 points)
        tempo_score = 0
        if 1.5 <= rep_duration <= 4.0: tempo_score = 30
        elif rep_duration < 1.5: tempo_score = 30 * (rep_duration / 1.5)
        else: tempo_score = 30 * max(0, (1 - (rep_duration - 4.0) / 2.0))

        total_score = int(depth_score + posture_score + tempo_score)
        breakdown = {"Depth": int(depth_score), "Posture": int(posture_score), "Tempo": int(tempo_score)}
        return total_score, breakdown

class LungeScorer(BaseScorer):
    def calculate_score(self, metrics):
        min_knee_angle = metrics.get('min_knee_angle', 180)
        max_torso_tilt = metrics.get('max_torso_tilt', 0)
        rep_duration = metrics.get('duration', 0)

        depth_score = 0
        if min_knee_angle < 110: depth_score = 35
        elif min_knee_angle > 130: depth_score = 0
        else: depth_score = 35 * (1 - (min_knee_angle - 110) / 20)

        posture_score = 0
        if max_torso_tilt < 25: posture_score = 35
        elif max_torso_tilt > 50: posture_score = 0
        else: posture_score = 35 * (1 - (max_torso_tilt - 25) / 25)

        tempo_score = 0
        if 1.2 <= rep_duration <= 3.5: tempo_score = 30
        elif rep_duration < 1.2: tempo_score = 30 * (rep_duration / 1.2)
        else: tempo_score = 30 * max(0, (1 - (rep_duration - 3.5) / 2.5))

        total_score = int(depth_score + posture_score + tempo_score)
        breakdown = {"Depth": int(depth_score), "Posture": int(posture_score), "Tempo": int(tempo_score)}
        return total_score, breakdown

class PushUpScorer(BaseScorer):
    def calculate_score(self, metrics):
        min_elbow_angle = metrics.get('min_knee_angle', 180) 
        max_back_sag = metrics.get('max_torso_tilt', 0)
        rep_duration = metrics.get('duration', 0)

        depth_score = 0
        if min_elbow_angle < 90: depth_score = 40
        elif min_elbow_angle > 120: depth_score = 0
        else: depth_score = 40 * (1 - (min_elbow_angle - 90) / 30)

        alignment_score = 0
        if max_back_sag < 15: alignment_score = 40
        elif max_back_sag > 35: alignment_score = 0
        else: alignment_score = 40 * (1 - (max_back_sag - 15) / 20)

        tempo_score = 0
        if 1.0 <= rep_duration <= 3.0: tempo_score = 20
        elif rep_duration < 1.0: tempo_score = 20 * (rep_duration / 1.0)
        else: tempo_score = 20 * max(0, (1 - (rep_duration - 3.0) / 2.0))

        total_score = int(depth_score + alignment_score + tempo_score)
        breakdown = {"Depth": int(depth_score), "Alignment": int(alignment_score), "Tempo": int(tempo_score)}
        return total_score, breakdown

class CurlScorer(BaseScorer):
    def calculate_score(self, metrics):
        """
        Professional Biomechanical Scoring for Curls.
        ROM (40%): Full contraction (elbow angle < 45)
        Stability (30%): Elbow pinning (horizontal drift < 0.12)
        Posture (30%): Torso swing (tilt angle < 10)
        """
        min_elbow_angle = metrics.get('min_elbow_angle', 180)
        max_drift = metrics.get('max_drift', 0)
        max_tilt = metrics.get('max_tilt', 0) # In degrees
        rep_duration = metrics.get('duration', 0)

        # 1. ROM Score (40 points)
        rom_score = 0
        if min_elbow_angle < 45: rom_score = 40
        elif min_elbow_angle > 110: rom_score = 0
        else: rom_score = 40 * (1 - (min_elbow_angle - 45) / 65)

        # 2. Stability Score (30 points) - Elbow Drift (Normalized)
        stability_score = 0
        if max_drift < 0.12: stability_score = 30
        elif max_drift > 0.30: stability_score = 0
        else: stability_score = 30 * (1 - (max_drift - 0.12) / 0.18)

        # 3. Posture Score (30 points) - Torso Tilt (Degrees)
        posture_score = 0
        if max_tilt < 10: posture_score = 30
        elif max_tilt > 25: posture_score = 0
        else: posture_score = 30 * (1 - (max_tilt - 10) / 15)

        # Total Calculation
        total_base = rom_score + stability_score + posture_score
        
        # 4. Tempo Penalty (Flat deduction)
        # Softened penalties: prioritize control, no penalty for slow reps
        penalty = 0
        if rep_duration < 0.7: # Extremely fast
            penalty = 20
        elif rep_duration < 1.0: # A bit fast
            penalty = 10

        final_score = int(max(0, total_base - penalty))
        
        breakdown = {
            "ROM": int(rom_score),
            "Stability": int(stability_score),
            "Posture": int(posture_score),
            "Penalty": int(penalty)
        }
        return final_score, breakdown
