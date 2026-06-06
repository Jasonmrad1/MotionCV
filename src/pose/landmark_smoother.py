import numpy as np

class LandmarkSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to landmarks.
    alpha: Smoothing factor (0 to 1). Higher = more reactive, Lower = smoother.
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_landmarks = {}

    def smooth(self, landmark_id, current_coords):
        if current_coords is None:
            return tuple(self.smoothed_landmarks[landmark_id].astype(int)) if landmark_id in self.smoothed_landmarks else None

        if landmark_id not in self.smoothed_landmarks:
            self.smoothed_landmarks[landmark_id] = np.array(current_coords)
            return (int(current_coords[0]), int(current_coords[1]))

        prev_smoothed = self.smoothed_landmarks[landmark_id]
        current_coords = np.array(current_coords)
        
        # EMA Formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        new_smoothed = self.alpha * current_coords + (1 - self.alpha) * prev_smoothed
        self.smoothed_landmarks[landmark_id] = new_smoothed
        
        # Return as integer tuple for OpenCV compatibility
        return (int(new_smoothed[0]), int(new_smoothed[1]))

    def reset(self):
        self.smoothed_landmarks = {}
