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
        if min_knee_angle < 80:
            depth_score = 30
        elif min_knee_angle > 110:
            depth_score = 0
        else:
            depth_score = 30 * (1 - (min_knee_angle - 80) / 30)

        # 2. Posture Score (40 points)
        posture_score = 0
        if max_torso_tilt < 30:
            posture_score = 40
        elif max_torso_tilt > 60:
            posture_score = 0
        else:
            posture_score = 40 * (1 - (max_torso_tilt - 30) / 30)

        # 3. Tempo Score (30 points)
        tempo_score = 0
        if 1.5 <= rep_duration <= 4.0:
            tempo_score = 30
        elif rep_duration < 1.5:
            tempo_score = 30 * (rep_duration / 1.5)
        else:
            tempo_score = 30 * max(0, (1 - (rep_duration - 4.0) / 2.0))

        total_score = int(depth_score + posture_score + tempo_score)
        
        breakdown = {
            "Depth": int(depth_score),
            "Posture": int(posture_score),
            "Tempo": int(tempo_score)
        }
        
        return total_score, breakdown
