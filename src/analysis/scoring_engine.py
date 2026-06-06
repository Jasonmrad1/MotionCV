class ScoringEngine:
    def __init__(self):
        pass

    def calculate_score(self, min_knee_angle, max_torso_tilt, rep_duration):
        """
        Calculates a score (0-100) based on depth, posture, and tempo.
        """
        # 1. Depth Score (30 points)
        # Target: < 80 degrees for full points. > 110 degrees for 0 points.
        depth_score = 0
        if min_knee_angle < 80:
            depth_score = 30
        elif min_knee_angle > 110:
            depth_score = 0
        else:
            depth_score = 30 * (1 - (min_knee_angle - 80) / 30)

        # 2. Posture Score (40 points)
        # Target: < 20 degrees torso tilt for full points. > 50 degrees for 0 points.
        posture_score = 0
        if max_torso_tilt < 20:
            posture_score = 40
        elif max_torso_tilt > 50:
            posture_score = 0
        else:
            posture_score = 40 * (1 - (max_torso_tilt - 20) / 30)

        # 3. Tempo Score (30 points)
        # Target: 2-4 seconds for a controlled rep.
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
