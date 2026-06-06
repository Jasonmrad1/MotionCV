import time

class SquatCounter:
    def __init__(self, min_visibility=0.5, buffer_limit=10): # Harder visibility requirement
        self.count = 0
        self.state = "UP"
        
        # Biomechanical "Sweet Spot" (Strict but fair)
        self.down_threshold = 90  # Target: Parallel
        self.up_threshold = 160   # Target: Full lockout
        self.min_rom = 60         # Require significant motion (e.g. 170 -> 90 -> 170)
        
        # Robustness Settings
        self.min_visibility = min_visibility
        self.invalid_frame_count = 0
        self.buffer_limit = buffer_limit
        
        # Persistence
        self.min_angle_reached = 180
        self.rep_start_time = 0
        self.is_active_squat = False

    def get_best_side_confidence(self, landmarks, l_indices, r_indices):
        """
        SWEET SPOT VISIBILITY: Requires BOTH Hip and Knee to be clearly present.
        """
        def calculate_leg_confidence(indices):
            hip_v = landmarks[indices[0]].visibility
            knee_v = landmarks[indices[1]].visibility
            # Hard cutoff: both must be clearly detected
            if hip_v >= self.min_visibility and knee_v >= self.min_visibility:
                return (hip_v + knee_v) / 2
            return 0

        l_conf = calculate_leg_confidence(l_indices)
        r_conf = calculate_leg_confidence(r_indices)
        
        best_v = max(l_conf, r_conf)
        side = "L" if l_conf >= r_conf else "R"
        
        if best_v >= 0.8: # Very high confidence
            return side, "HIGH"
        elif best_v >= self.min_visibility:
            return side, "MEDIUM"
        else:
            return side, "LOW"

    def update(self, knee_angle, confidence_level):
        """
        The "Biomechanical Sweet Spot" State Machine.
        """
        rep_completed = False
        duration = 0
        
        if confidence_level == "LOW":
            self.invalid_frame_count += 1
            self.is_active_squat = False
            return self.count, self.state, False, False, False, 0

        self.invalid_frame_count = 0 

        # 1. Intent Detection (Active Gate)
        # Active only if actually squatting
        if knee_angle < 145:
            self.is_active_squat = True
        else:
            if self.state == "UP":
                self.is_active_squat = False

        # 2. Resilient State Machine
        if self.state == "UP":
            if knee_angle < self.down_threshold:
                self.state = "DOWN"
                self.rep_start_time = time.time()
                self.min_angle_reached = knee_angle
        
        elif self.state == "DOWN":
            if knee_angle < self.min_angle_reached:
                self.min_angle_reached = knee_angle
                
            if knee_angle > self.up_threshold:
                # 3. Validation: The Sweet Spot
                rom = self.up_threshold - self.min_angle_reached
                duration = time.time() - self.rep_start_time
                
                # REPS must be full movement and controlled duration
                if rom >= self.min_rom and 0.8 < duration < 4.0:
                    self.count += 1
                    rep_completed = True
                
                self.state = "UP"
                self.min_angle_reached = 180
        
        return self.count, self.state, True, rep_completed, self.is_active_squat, duration
