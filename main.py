import cv2
import time
import numpy as np
from src.pose.pose_detector import PoseDetector
from src.exercises.squat import Squat
from src.analytics.workout_summary import WorkoutSummary
from src.ui.overlays import draw_hud

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PoseDetector(model_path="models/pose_landmarker_lite.task")
    exercise = Squat()
    summary = WorkoutSummary()
    
    prev_time = time.time()
    last_score = None

    while True:
        success, img = cap.read()
        if not success: break

        h, w, _ = img.shape
        timestamp_ms = int(time.time() * 1000)
        result = detector.find_pose(img, timestamp_ms)

        reps, state, is_active, feedback, warning_msg = 0, "UP", False, [], None
        side = None

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            reps, state, is_active, feedback, warning_msg = exercise.process(landmarks, w, h, timestamp_ms)
            
            metrics = exercise.get_metrics()
            last_score = metrics["last_score"]
            side = metrics["side"]

            if exercise.counter.last_rep_data and exercise.counter.last_rep_data["valid"]:
                summary.add_rep(last_score, {}, feedback)

            img = detector.draw_landmarks(img, result)

        # UI & Performance
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        draw_hud(img, reps, state, fps, feedback, is_active, last_score, warning_msg, side)

        cv2.imshow("MotionCV - Coaching & Scoring", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
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
