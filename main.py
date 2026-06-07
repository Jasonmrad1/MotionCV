import cv2
import time
import numpy as np
from src.pose.pose_detector import PoseDetector
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle
from src.analysis.rep_counter import SquatCounter
from src.analysis.form_analyzer import FormAnalyzer
from src.analysis.scoring_engine import ScoringEngine
from src.analytics.workout_summary import WorkoutSummary

def draw_hud(img, reps, state, fps, feedback_list, is_active, last_score=None, warning_msg=None):
    h, w, _ = img.shape
    
    # Header bar
    if warning_msg:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, warning_msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(img, "Setup: Side View | Hip Height | 6ft Distance", (w - 450, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Sidebar
    panel_w = 300
    panel_h = 320
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 60), (panel_w, 60 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Status colors
    active_color = (0, 255, 0) if is_active else (150, 150, 150)
    rep_color = (0, 255, 0) if not warning_msg else (0, 0, 255)
    
    cv2.putText(img, f"REPS: {reps}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rep_color, 3)
    cv2.putText(img, f"STATE: {state}", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2)
    
    if last_score is not None:
        score_color = (0, 255, 0) if last_score > 80 else (0, 165, 255) if last_score > 60 else (0, 0, 255)
        cv2.putText(img, f"SCORE: {last_score}", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)

    cv2.putText(img, f"SQUAT ACTIVE" if is_active else "IDLE", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1)
    
    cv2.putText(img, "LATEST FEEDBACK:", (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for i, feedback in enumerate(feedback_list):
        fb_color = (0, 255, 0) if any(word in feedback.lower() for word in ["excellent", "great", "target", "neutral", "stable"]) else (0, 165, 255)
        cv2.putText(img, f"- {feedback}", (25, 280 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PoseDetector(model_path="models/pose_landmarker_lite.task")
    counter = SquatCounter(min_visibility=0.15, buffer_limit=30)
    analyzer = FormAnalyzer()
    scorer = ScoringEngine()
    summary = WorkoutSummary()
    smoother = LandmarkSmoother(alpha=0.6)
    
    prev_time = time.time()
    # SQUAT KEYPOINT MAP
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    
    L_INDICES = [L_HIP, L_KNEE, L_ANKLE]
    R_INDICES = [R_HIP, R_KNEE, R_ANKLE]
    KEY_JOINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

    last_valid_coords = {idx: None for idx in KEY_JOINTS}
    latest_rep_summary = ["Ready to start"]
    last_score = None

    while True:
        success, img = cap.read()
        if not success: break

        h, w, _ = img.shape
        timestamp_ms = int(time.time() * 1000)
        result = detector.find_pose(img, timestamp_ms)

        warning_msg = None
        reps, state, is_active = counter.count, counter.state, counter.is_active_squat

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            # 1. HARD LEG LOCK: Dynamic Side Selection based on REAL visibility
            side_prefix, conf_level = counter.get_best_side_confidence(landmarks, L_INDICES, R_INDICES)
            
            # Select joints based on best side
            HIP = L_HIP if side_prefix == "L" else R_HIP
            KNEE = L_KNEE if side_prefix == "L" else R_KNEE
            ANKLE = L_ANKLE if side_prefix == "L" else R_ANKLE
            SHOULDER = L_SHOULDER if side_prefix == "L" else R_SHOULDER

            def get_target_pt(idx, is_leg=True):
                # Persistence is only for BRIEF occlusions (300ms)
                # If current visibility is low, we don't 'trust' the coordinate for rep counting
                current_v = landmarks[idx].visibility
                if current_v >= 0.4:
                    coords = (landmarks[idx].x * w, landmarks[idx].y * h)
                    last_valid_coords[idx] = coords
                    return coords
                # For legs, if visibility is very low, return None to signal 'Not found'
                if is_leg and current_v < 0.15:
                    return None
                return last_valid_coords[idx]

            # 2. Extract Targeted Points - NO PERSISTENCE if legs are truly gone
            raw_hip = get_target_pt(HIP, is_leg=True)
            raw_knee = get_target_pt(KNEE, is_leg=True)
            raw_ankle = get_target_pt(ANKLE, is_leg=True)
            shoulder = get_target_pt(SHOULDER, is_leg=False)

            # Leg smoothing
            hip = smoother.smooth(HIP, raw_hip)
            knee = smoother.smooth(KNEE, raw_knee)
            ankle = smoother.smooth(ANKLE, raw_ankle)

            if hip and knee and ankle and conf_level != "LOW":
                angle = compute_angle(hip, knee, ankle)
                
                # 3. Confidence-Aware Update
                reps, state, is_viable, rep_done, is_active, duration = counter.update(angle, conf_level, timestamp_ms / 1000.0)

                if is_viable:
                    if is_active and shoulder:
                        realtime_cues = analyzer.analyze_posture(hip, shoulder, knee, ankle, angle, is_active)
                        for i, cue in enumerate(realtime_cues):
                            cv2.putText(img, cue, (int(w/2) - 100, 100 + i*30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                    if rep_done:
                        feedback, min_angle, max_tilt = analyzer.get_rep_summary()
                        latest_rep_summary = feedback
                        total_score, _ = scorer.calculate_score(min_angle, max_tilt, duration)
                        last_score = total_score
                        summary.add_rep(total_score, {}, feedback)
                        analyzer.reset_rep_metrics()

                    img = detector.draw_landmarks(img, result)
                    
                    # HUD Status updates
                    if conf_level == "MEDIUM":
                        warning_msg = f"REDUCED VISIBILITY (SIDE: {side_prefix})"
                    elif side_prefix:
                        # Subtle indicator of which side we're tracking
                        cv2.putText(img, f"SIDE: {side_prefix}", (w - 100, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                else:
                    warning_msg = "POSE LOST - ADJUST POSITION"
            else:
                warning_msg = "INITIALIZING CORE JOINTS..."
        else:
            warning_msg = "NO PERSON DETECTED"

        # UI & Performance
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        draw_hud(img, reps, state, fps, latest_rep_summary, is_active, last_score, warning_msg)

        cv2.imshow("MotionCV - Coaching & Scoring", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            # Show final summary on exit
            final_stats = summary.get_summary()
            if final_stats:
                print("\n=== WORKOUT SUMMARY ===")
                print(f"Total Reps: {final_stats['total_reps']}")
                print(f"Average Score: {final_stats['average_score']}")
                print(f"Top Focus Areas: {', '.join(final_stats['top_cues'])}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
