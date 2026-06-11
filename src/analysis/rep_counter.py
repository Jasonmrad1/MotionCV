import time
from abc import ABC, abstractmethod

class BaseRepCounter(ABC):
    def __init__(self, min_visibility=0.1, buffer_limit=30, k_frames=3):
        self.count = 0
        self.state = "UP"
        self.is_active = False
        self.last_rep_data = None
        self.min_visibility = min_visibility
        self.buffer_limit = buffer_limit
        self.visibility_buffer = []
        self.start_time = None
        
        # State Transition Hardening
        self.k_frames = k_frames
        self.state_frames = 0
        self.pending_state = "UP"

    @abstractmethod
    def update(self, signal_value, visibility_score, timestamp):
        pass

    def _check_visibility(self, visibility_score):
        v_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
        v_val = v_map.get(visibility_score, 0.0)
        
        self.visibility_buffer.append(v_val)
        if len(self.visibility_buffer) > self.buffer_limit:
            self.visibility_buffer.pop(0)
            
        avg_v = sum(self.visibility_buffer) / len(self.visibility_buffer)
        return avg_v >= self.min_visibility

class AngleCounter(BaseRepCounter):
    """
    Generic counter for exercises driven by joint angles (Squat, Lunge, Curl).
    """
    def __init__(self, down_thresh, up_thresh, target_depth, min_rom=30, min_visibility=0.1, k_frames=3):
        super().__init__(min_visibility=min_visibility, k_frames=k_frames)
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.target_depth = target_depth
        self.min_rom = min_rom
        self.min_angle_reached = 180

    def update(self, angle, visibility_score, timestamp):
        is_viable = self._check_visibility(visibility_score)
        rep_completed = False
        duration = 0

        if not is_viable:
            return self.count, self.state, False, False, False, 0

        if self.state == "UP":
            if angle < self.down_thresh:
                if self.pending_state != "DOWN":
                    self.pending_state = "DOWN"
                    self.state_frames = 1
                else:
                    self.state_frames += 1
                    if self.state_frames >= self.k_frames:
                        self.state = "DOWN"
                        self.start_time = timestamp
                        self.min_angle_reached = angle
                        self.is_active = True
            else:
                self.pending_state = "UP"
                self.state_frames = 0
        
        elif self.state == "DOWN":
            self.min_angle_reached = min(self.min_angle_reached, angle)
            
            if angle > self.up_thresh:
                if self.pending_state != "UP":
                    self.pending_state = "UP"
                    self.state_frames = 1
                else:
                    self.state_frames += 1
                    if self.state_frames >= self.k_frames:
                        duration = timestamp - self.start_time
                        rom = self.up_thresh - self.min_angle_reached
                        
                        if rom >= self.min_rom and duration > 0.4:
                            self.count += 1
                            rep_completed = True
                            self.last_rep_data = {
                                "min_angle": self.min_angle_reached,
                                "duration": duration,
                                "rom": rom
                            }
                        
                        self.state = "UP"
                        self.is_active = False
                        self.min_angle_reached = 180
            else:
                self.pending_state = "DOWN"
                self.state_frames = 0

        return self.count, self.state, True, rep_completed, self.is_active, duration

class TravelCounter(BaseRepCounter):
    """
    Generic counter for exercises driven by vertical travel (Push-Up, Split Squat).
    """
    def __init__(self, descent_thresh_pct=0.08, return_thresh_pct=0.04, min_duration=0.15, min_visibility=0.1, k_frames=3):
        super().__init__(min_visibility=min_visibility, k_frames=k_frames)
        self.descent_thresh_pct = descent_thresh_pct
        self.return_thresh_pct = return_thresh_pct
        self.min_duration = min_duration
        self.baseline_y = None
        self.peak_descent = 0

    def update(self, y_value, torso_len, visibility_score, timestamp):
        is_viable = self._check_visibility(visibility_score)
        rep_completed = False
        duration = 0

        if not is_viable or torso_len <= 0:
            return self.count, self.state, False, False, False, 0

        if self.baseline_y is None:
            self.baseline_y = y_value
            return self.count, self.state, True, False, False, 0

        descent_ratio = (y_value - self.baseline_y) / torso_len

        if self.state == "UP":
            if descent_ratio > self.descent_thresh_pct:
                if self.pending_state != "DOWN":
                    self.pending_state = "DOWN"
                    self.state_frames = 1
                else:
                    self.state_frames += 1
                    if self.state_frames >= self.k_frames:
                        self.state = "DOWN"
                        self.start_time = timestamp
                        self.peak_descent = descent_ratio
                        self.is_active = True
            else:
                self.pending_state = "UP"
                self.state_frames = 0
                if descent_ratio < (self.return_thresh_pct / 2):
                    self.baseline_y = 0.9 * self.baseline_y + 0.1 * y_value
                    self.is_active = False

        elif self.state == "DOWN":
            self.peak_descent = max(self.peak_descent, descent_ratio)
            
            if descent_ratio < self.return_thresh_pct:
                if self.pending_state != "UP":
                    self.pending_state = "UP"
                    self.state_frames = 1
                else:
                    self.state_frames += 1
                    if self.state_frames >= self.k_frames:
                        duration = timestamp - self.start_time
                        
                        if duration > self.min_duration and self.peak_descent > (self.descent_thresh_pct * 1.5):
                            self.count += 1
                            rep_completed = True
                        
                        self.state = "UP"
                        self.is_active = False
                        self.peak_descent = 0
            else:
                self.pending_state = "DOWN"
                self.state_frames = 0

        return self.count, self.state, True, rep_completed, self.is_active, duration
