import time

class SquatCounter:
    def __init__(self, min_visibility=0.5, buffer_limit=10): # Harder visibility requirement
        self.count = 0
        self.state = "UP"
        # Olympic sets
        # # Biomechanical "Sweet Spot" (Strict but fair)
        # self.down_threshold = 90  # Target: Parallel
        # self.up_threshold = 160   # Target: Full lockout
        # self.min_rom = 60         # Require significant motion (e.g. 170 -> 90 -> 170)

        # ✔ FIXED (slightly more realistic, still strict)
        self.down_threshold = 110   # was 100 → enter bottom phase earlier
        self.up_threshold = 155     # was 160 → too high for imperfect extension
        self.min_rom = 45           # was 60 → too strict, rejects valid reps
        self.target_depth = 95      # relaxed from 90 to 95 for better user experience
        
        # Robustness Settings
        self.min_visibility = min_visibility
        self.invalid_frame_count = 0
        self.buffer_limit = buffer_limit
        
        # Persistence
        self.min_angle_reached = 180
        self.rep_start_time = 0
        self.is_active_squat = False
        self.last_rep_data = None # Stores {status, rom, duration, min_angle}

    def update(self, knee_angle, confidence_level, timestamp=None):
        """
        The "Biomechanical Sweet Spot" State Machine.
        """
        rep_completed = False
        duration = 0
        curr_time = timestamp if timestamp is not None else time.time()
        
        # Reset per-frame attempt data
        self.last_rep_data = None
        
        if confidence_level == "LOW":
            self.invalid_frame_count += 1
            self.is_active_squat = False
            return self.count, self.state, False, False, False, 0

        self.invalid_frame_count = 0 

        # 1. Intent Detection (Active Gate)
        # Active only if actually squatting (Relaxed from 145 to 160)
        if knee_angle < 160:
            self.is_active_squat = True
        else:
            if self.state == "UP":
                self.is_active_squat = False

        # 2. Resilient State Machine
        if self.state == "UP":
            if knee_angle < self.down_threshold:
                self.state = "DOWN"
                self.rep_start_time = curr_time
                self.min_angle_reached = knee_angle
        
        elif self.state == "DOWN":
            if knee_angle < self.min_angle_reached:
                self.min_angle_reached = knee_angle
                
            if knee_angle > self.up_threshold:
                # 3. Validation: The Sweet Spot
                rom = self.up_threshold - self.min_angle_reached
                duration = curr_time - self.rep_start_time
                
                # REPS must be full movement and controlled duration
                is_valid = True
                reason = "VALID"
                
                if rom < self.min_rom:
                    is_valid = False
                    reason = f"ROM TOO LOW ({int(rom)} < {self.min_rom})"
                elif self.min_angle_reached > self.target_depth:
                    is_valid = False
                    reason = f"SHALLOW DEPTH ({int(self.min_angle_reached)} > {self.target_depth})"
                elif duration <= 0.4:
                    is_valid = False
                    reason = f"TOO FAST ({duration:.2f}s <= 0.4s)"
                elif duration >= 4.0:
                    is_valid = False
                    reason = f"TOO SLOW ({duration:.2f}s >= 4.0s)"

                self.last_rep_data = {
                    "valid": is_valid,
                    "reason": reason,
                    "rom": int(rom),
                    "duration": round(duration, 2),
                    "min_angle": int(self.min_angle_reached)
                }

                if is_valid:
                    self.count += 1
                    rep_completed = True
                
                self.state = "UP"
                self.min_angle_reached = 180
        
        return self.count, self.state, True, rep_completed, self.is_active_squat, duration
