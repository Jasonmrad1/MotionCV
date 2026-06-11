import numpy as np
import time

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        
    def update_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def smooth(self, x, dt):
        if dt <= 0: return x if self.x_prev is None else self.x_prev
        self.freq = 1.0 / dt
        
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.d_cutoff) * dx + (1 - self._alpha(self.d_cutoff)) * self.dx_prev
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        alpha = self._alpha(cutoff)

        result = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = result
        return result

class LandmarkSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.05, deadband_epsilon=0.002):
        self.filters = {}
        self.base_min_cutoff = min_cutoff
        self.base_beta = beta
        self.deadband_epsilon = deadband_epsilon
        self.last_time = None
        self.smoothed_landmarks = {}
        self.current_dt = 1.0 / 30.0

    def start_frame(self):
        """Must be called at the beginning of each frame processing."""
        now = time.time()
        if self.last_time is not None:
            self.current_dt = max(now - self.last_time, 0.001)
        self.last_time = now

    def smooth(self, landmark_id, current_coords):
        coords = np.array(current_coords)
        if landmark_id not in self.filters:
            self.filters[landmark_id] = OneEuroFilter(1.0/self.current_dt, 
                                                      min_cutoff=self.base_min_cutoff, 
                                                      beta=self.base_beta)
            self.smoothed_landmarks[landmark_id] = coords
            return tuple(coords)

        # 1. Adaptive Filtering based on previous velocity estimate
        prev_coords = self.smoothed_landmarks.get(landmark_id, coords)
        velocity = np.linalg.norm(coords - prev_coords) / self.current_dt
        
        # Heuristic: Fast movement = high reactivity
        # Slow movement/hold = high smoothing
        adaptive_cutoff = self.base_min_cutoff
        adaptive_beta = self.base_beta
        
        if velocity > 1.5:
            adaptive_cutoff = min(3.0, self.base_min_cutoff * 2.0)
            adaptive_beta = min(0.2, self.base_beta * 2.0)
        elif velocity < 0.2:
            adaptive_cutoff = max(0.1, self.base_min_cutoff * 0.5)
            adaptive_beta = max(0.01, self.base_beta * 0.5)

        self.filters[landmark_id].update_params(adaptive_cutoff, adaptive_beta)

        # 2. Smooth
        smoothed = self.filters[landmark_id].smooth(coords, self.current_dt)
        
        # 3. Anti-Jitter Deadband
        # If the change is tiny (e.g. < 0.002 normalized units), ignore it to lock the joint
        diff = np.linalg.norm(smoothed - prev_coords)
        if diff < self.deadband_epsilon:
            smoothed = prev_coords # Freeze joint

        self.smoothed_landmarks[landmark_id] = smoothed
        return tuple(smoothed)

    def reset(self):
        self.filters = {}
        self.smoothed_landmarks = {}
        self.last_time = None
