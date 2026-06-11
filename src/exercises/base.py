from abc import ABC, abstractmethod

class BaseExercise(ABC):
    def __init__(self):
        self.reps = 0
        self.state = "UP"
        self.is_active = False
        self.latest_feedback = []
        self.last_score = None
        self.warning_msg = None

    @abstractmethod
    def process(self, landmarks, world_landmarks, w, h, timestamp_ms, img=None):
        """
        Processes pose landmarks and world landmarks (metric 3D) and updates exercise state.
        Returns: (reps, state, is_active, feedback, warning, rep_done)
        """
        pass

    @abstractmethod
    def get_metrics(self):
        """
        Returns a dictionary of current metrics.
        """
        pass

    def get_best_side(self, landmarks, l_indices, r_indices, min_visibility=0.1):
        """
        Calculates which side (Left or Right) has better visibility for the required joints.
        """
        def calculate_confidence(indices):
            visibilities = [landmarks[idx].visibility for idx in indices]
            # Use average visibility instead of strict 'all' check
            avg_v = sum(visibilities) / len(visibilities)
            # If any joint is completely missing (< 0.05), fail the side
            if any(v < 0.05 for v in visibilities):
                return 0
            return avg_v

        l_conf = calculate_confidence(l_indices)
        r_conf = calculate_confidence(r_indices)
        
        best_v = max(l_conf, r_conf)
        side = "L" if l_conf >= r_conf else "R"
        
        if best_v >= 0.7: # Relaxed from 0.8
            return side, "HIGH"
        elif best_v >= min_visibility:
            return side, "MEDIUM"
        else:
            return side, "LOW"
