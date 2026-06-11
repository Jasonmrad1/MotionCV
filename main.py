import cv2
import time
import argparse
import numpy as np
from src.pose.pose_detector import PoseDetector
from src.exercises import get_exercise_class, get_available_exercises
from src.analytics.workout_summary import WorkoutSummary
from src.ui.overlays import draw_hud, draw_session_report

def select_exercise_interactively():
    available = get_available_exercises()
    
    print("\n" + "="*30)
    print("      MOTIONCV - SELECT EXERCISE")
    print("="*30)
    for i, ex in enumerate(available, 1):
        print(f"  {i}. {ex.capitalize()}")
    print("="*30)
    
    while True:
        choice = input(f"\nSelect exercise (1-{len(available)}) or type name: ").strip().lower()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available): return available[idx]
        if choice in available: return choice
        print("Invalid choice.")

def main():
    parser = argparse.ArgumentParser(description="MotionCV AI Fitness Coach")
    parser.add_argument("--ex", type=str, default=None, choices=["squats", "lunges"],
                        help="Select exercise: squats or lunges")
    args = parser.parse_args()
    
    exercise_name = args.ex if args.ex else select_exercise_interactively()
    print(f"\nInitializing {exercise_name.capitalize()} Session...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = PoseDetector()
    exercise_cls = get_exercise_class(exercise_name)
    exercise = exercise_cls()
    summary = WorkoutSummary()
    
    prev_time = time.time()
    last_score = None

    while True:
        success, img = cap.read()
        if not success: break

        timestamp_ms = int(time.time() * 1000)
        # Use previous frame's active status to harden detection
        is_active_rep = exercise.is_active if hasattr(exercise, 'is_active') else False
        result = detector.find_pose(img, timestamp_ms, is_active_rep=is_active_rep)

        reps, state, is_active, feedback, warning_msg = 0, "UP", False, [], None
        side = None

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            reps, state, is_active, feedback, warning_msg, rep_done = exercise.process(
                landmarks, world_landmarks, img.shape[1], img.shape[0], timestamp_ms, img=img
            )
            
            metrics = exercise.get_metrics()
            last_score = metrics["last_score"]
            side = metrics["side"]
            current_angle = metrics["angle"]

            if rep_done:
                summary.add_rep(last_score, {}, feedback, current_angle)

            img = detector.draw_landmarks(img, result, joints_to_draw=exercise.KEY_JOINTS)

        # UI & Performance
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        draw_hud(img, reps, state, fps, feedback, is_active, last_score, warning_msg, side, exercise_name)

        cv2.imshow("MotionCV - Coaching & Scoring", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            final_stats = summary.get_summary()
            if final_stats:
                report_img = img.copy()
                draw_session_report(report_img, final_stats)
                cv2.imshow("MotionCV - Coaching & Scoring", report_img)
                cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
