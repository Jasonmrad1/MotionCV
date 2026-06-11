import cv2
import mediapipe as mp
import numpy as np
import logging
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class TrackingState:
    LOST = "LOST"
    SEARCHING = "SEARCHING"
    REACQUIRING = "REACQUIRING"
    LOCKED = "LOCKED"

class PoseDetector:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=3, # Support multi-person for filtering
            min_pose_detection_confidence=0.4, # Lowered from 0.5 for side-view robustness
            min_pose_presence_confidence=0.4,  # Lowered from 0.5
            min_tracking_confidence=0.4,        # Lowered from 0.5
            output_segmentation_masks=False
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # Tracking FSM State
        self.state = TrackingState.LOST
        self.confidence = 0.0
        self.last_centroid = None
        self.last_velocity = np.array([0.0, 0.0])
        self.consecutive_matches = 0
        self.lost_frames = 0
        self.jump_frames = 0 # Hysteresis for sudden target jumps
        
        # Tuning parameters
        self.DECAY_FACTOR = 0.85
        self.LOCK_THRESHOLD = 4 # Frames required to lock
        self.LOST_THRESHOLD = 10 # Increased from 5 to be stickier
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("Tracker")

    def _get_centroid(self, pose):
        try:
            # Expanded centroid: average of shoulders and hips
            l_sh, r_sh = pose[11], pose[12]
            l_hip, r_hip = pose[23], pose[24]
            return np.array([
                (l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0,
                (l_sh.y + r_sh.y + l_hip.y + r_hip.y) / 4.0
            ])
        except (IndexError, AttributeError):
            return None

    def _get_visibility_score(self, pose):
        # Average visibility of key torso and limb joints
        joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        score = 0
        count = 0
        for j in joints:
            try:
                score += pose[j].visibility
                count += 1
            except IndexError:
                pass
        return score / max(1, count)

    def find_pose(self, img, timestamp_ms, is_active_rep=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result or not result.pose_landmarks:
            self._handle_no_detection()
            return result

        best_pose_idx = -1
        max_score = -1.0
        
        for i, pose in enumerate(result.pose_landmarks):
            curr_centroid = self._get_centroid(pose)
            if curr_centroid is None: continue

            # 1. Area Component
            try:
                l_sh, r_sh = pose[11], pose[12]
                l_hip, r_hip = pose[23], pose[24]
                h = abs((l_sh.y + r_sh.y)/2 - (l_hip.y + r_hip.y)/2)
                w = abs(l_sh.x - r_sh.x)
                area = h * w
            except: area = 0.01

            # 2. Visibility Component
            vis_score = self._get_visibility_score(pose)

            # 3. Temporal Consistency
            temporal_score = 1.0
            if self.last_centroid is not None:
                dist = np.linalg.norm(curr_centroid - self.last_centroid)
                temporal_score = max(0.0, 1.0 - (dist / 0.4)) 
            
            # Weighted Total Score
            if self.state == TrackingState.LOCKED:
                # REJECTION LOGIC: If it's too small or too far, it's likely background noise
                if area < 0.005: continue # Too small to be a person
                if vis_score < 0.2: continue # Too blurry
                
                score = (area * 0.2) + (vis_score * 0.2) + (temporal_score * 0.6)
            else:
                score = (area * 0.4) + (vis_score * 0.4) + (temporal_score * 0.2)
            
            if score > max_score:
                max_score = score
                best_pose_idx = i

        if best_pose_idx != -1:
            pose = result.pose_landmarks[best_pose_idx]
            curr_centroid = self._get_centroid(pose)
            
            if self.last_centroid is not None:
                dist = np.linalg.norm(curr_centroid - self.last_centroid)
                
                # REP-LOCK: If in a rep, be 2x as skeptical of jumps
                jump_limit = 0.45 if not is_active_rep else 0.70
                confirm_limit = 3 if not is_active_rep else 6

                if dist > jump_limit:
                    self.jump_frames += 1
                    if self.jump_frames < confirm_limit:
                        self.logger.warning(f"Ignoring Potential Noise (dist: {dist:.2f})")
                        return None # Treat as no detection to avoid tracking the 'bed'
            
            self._update_state(pose, max_score)
            
            # Filter output
            result.pose_landmarks = [result.pose_landmarks[best_pose_idx]]
            if result.pose_world_landmarks:
                result.pose_world_landmarks = [result.pose_world_landmarks[best_pose_idx]]
        else:
            self._handle_no_detection()

        return result

    def _update_state(self, best_pose, current_score):
        curr_centroid = self._get_centroid(best_pose)
        
        # Update Temporal Confidence
        self.confidence = (self.confidence * self.DECAY_FACTOR) + current_score
        
        if self.last_centroid is not None:
            velocity = curr_centroid - self.last_centroid
            # Smooth velocity update
            self.last_velocity = (self.last_velocity * 0.5) + (velocity * 0.5)
            
            # Check if this was a confirmed jump
            dist = np.linalg.norm(curr_centroid - self.last_centroid)
            if dist > 0.45:
                self.logger.warning(f"Target Jump Confirmed (dist: {dist:.2f})")
                self.consecutive_matches = 0
                self.state = TrackingState.REACQUIRING
        else:
            self.last_velocity = np.array([0.0, 0.0])

        self.last_centroid = curr_centroid
        self.lost_frames = 0
        self.jump_frames = 0 # Reset jump frames on stable detection

        # State Transitions
        if self.state in [TrackingState.LOST, TrackingState.SEARCHING]:
            self.consecutive_matches += 1
            if self.consecutive_matches >= self.LOCK_THRESHOLD:
                self.state = TrackingState.LOCKED
                self.logger.info("Tracking LOCKED")
            else:
                self.state = TrackingState.SEARCHING
        
        elif self.state == TrackingState.REACQUIRING:
            # INSTANT RE-LOCK: If we dropped state but found the target in almost the exact same spot,
            # don't wait LOCK_THRESHOLD frames. Instantly lock.
            if 'dist' in locals() and dist < 0.15:
                self.consecutive_matches = self.LOCK_THRESHOLD
                self.state = TrackingState.LOCKED
                self.logger.info("Tracking INSTANTLY RE-LOCKED (Proximity)")
            else:
                self.consecutive_matches += 1
                if self.consecutive_matches >= self.LOCK_THRESHOLD:
                    self.state = TrackingState.LOCKED
                    self.logger.info("Tracking RE-LOCKED")

    def _handle_no_detection(self):
        self.consecutive_matches = 0
        self.lost_frames += 1
        self.jump_frames = 0
        
        # Decay confidence
        self.confidence *= self.DECAY_FACTOR
        
        # Slow down velocity prediction
        self.last_velocity *= 0.5
        if self.last_centroid is not None:
            self.last_centroid += self.last_velocity

        if self.state == TrackingState.LOCKED:
            if self.lost_frames >= self.LOST_THRESHOLD:
                self.state = TrackingState.LOST
                self.last_centroid = None
                self.last_velocity = np.array([0.0, 0.0])
                self.confidence = 0.0
                self.logger.info("Tracking LOST (Threshold Exceeded)")
            elif self.lost_frames > 3: # Stay LOCKED for up to 3 missing frames
                self.state = TrackingState.REACQUIRING
                self.logger.info(f"Tracking REACQUIRING (Lost {self.lost_frames} frames)")
            # If <= 3 frames, we stay in LOCKED state to prevent flickering UI/logic
                
        elif self.state == TrackingState.SEARCHING:
             if self.lost_frames >= self.LOST_THRESHOLD:
                 self.state = TrackingState.LOST
                 self.last_centroid = None

    def draw_landmarks(self, img, result, joints_to_draw=None):
        if not result or not result.pose_landmarks:
            return img

        h, w, _ = img.shape
        DEFAULT_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]

        active_joints = joints_to_draw if joints_to_draw is not None else DEFAULT_JOINTS
        landmarks = result.pose_landmarks[0]

        # Draw Skeleton
        for start_idx, end_idx in CONNECTIONS:
            is_torso = start_idx in [11, 12, 23, 24] and end_idx in [11, 12, 23, 24]
            if is_torso or (start_idx in active_joints and end_idx in active_joints):
                if landmarks[start_idx].visibility > 0.15 and landmarks[end_idx].visibility > 0.15:
                    pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(img, pt1, pt2, (180, 180, 180), 2)

        # Color dots based on State
        color = (0, 255, 0) # Green = LOCKED
        if self.state == TrackingState.SEARCHING: color = (0, 255, 255) # Yellow
        elif self.state == TrackingState.REACQUIRING: color = (0, 165, 255) # Orange

        for idx in active_joints:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.15:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (x, y), 4, color, -1)

        # Telemetry Text
        cv2.putText(img, f"STATE: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f"CONF: {self.confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img