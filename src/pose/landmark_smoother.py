import numpy as np

class LandmarkSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to landmarks.
    alpha: Smoothing factor (0 to 1). Higher = more reactive, Lower = smoother.
    max_jump: Maximum allowed pixel jump between frames before dampening.
    """
    def __init__(self, alpha=0.5, max_jump=50):
        self.alpha = alpha
        self.max_jump = max_jump
        self.smoothed_landmarks = {}

    def smooth(self, landmark_id, current_coords):
        if current_coords is None:
            return tuple(self.smoothed_landmarks[landmark_id].astype(int)) if landmark_id in self.smoothed_landmarks else None

        current_coords = np.array(current_coords)

        if landmark_id not in self.smoothed_landmarks:
            self.smoothed_landmarks[landmark_id] = current_coords
            return (int(current_coords[0]), int(current_coords[1]))

        prev_smoothed = self.smoothed_landmarks[landmark_id]
        
        # Calculate distance of the jump
        dist = np.linalg.norm(current_coords - prev_smoothed)
        
        # If jump is too large, it's likely noise. Dampen the alpha significantly.
        effective_alpha = self.alpha
        if dist > self.max_jump:
            effective_alpha = self.alpha * 0.1 # Heavily dampen the jump
        
        # EMA Formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        new_smoothed = effective_alpha * current_coords + (1 - effective_alpha) * prev_smoothed
        self.smoothed_landmarks[landmark_id] = new_smoothed
        
        # Return as integer tuple for OpenCV compatibility
        return (int(new_smoothed[0]), int(new_smoothed[1]))

    def reset(self):
        self.smoothed_landmarks = {}
