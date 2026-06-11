import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import csv
import json
import argparse
import numpy as np

from src.pose.pose_detector import PoseDetector
from src.exercises import get_exercise_class
from src.ui.overlays import draw_hud, draw_session_report

class BenchmarkRunner:
    def __init__(self, exercise="squats", save_video=False, live=False):
        self.exercise_name = exercise
        self.video_dir = f"videos/benchmark/{exercise}"
        self.results_file = "benchmark/benchmark_results.csv"
        self.logs_dir = "benchmark/logs"
        self.debug_dir = "benchmark/debug_videos"

        self.save_video = save_video
        self.live = live

        self.headers = [
            "video_name", "expected_reps", "detected_reps", "error",
            "avg_fps", "total_frames",
            "pose_coverage_pct", "avg_visibility", "max_pose_loss_streak",
            "active_frames", "active_coverage_pct",
            "down_transitions", "up_transitions", "aborted_squats"
        ]

        self.ground_truth = self.load_ground_truth()

    def load_ground_truth(self):
        gt = {}
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("expected_reps") and row.get("video_name"):
                            gt[row["video_name"]] = int(row["expected_reps"])
            except:
                pass
        return gt

    def run_benchmark(self):
        video_files = [f for f in os.listdir(self.video_dir)
                       if f.endswith(('.mp4', '.mov', '.avi'))]

        if not video_files:
            print(f"No videos found in {self.video_dir}")
            return

        results = []

        for video_file in video_files:
            print(f"\nProcessing: {video_file}...")

            result = self.process_video(
                os.path.join(self.video_dir, video_file),
                video_file
            )

            if result:
                results.append(result)

                print(f"  Detected Reps: {result['detected_reps']}")
                print(f"  Aborted Squats: {result['aborted_squats']}")
                print(f"  Pose Coverage: {result['pose_coverage_pct']:.1f}%")
                print(f"  Max Loss Streak: {result['max_pose_loss_streak']} frames")

        self.save_results(results)

    def process_video(self, path, filename):
        detector = PoseDetector(model_path="models/pose_landmarker_lite.task")
        cap = cv2.VideoCapture(path)

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in <= 0:
            fps_in = 30

        ret, test_frame = cap.read()
        if not ret:
            print(f"[ERROR] Cannot read video: {filename}")
            return None

        height, width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        out = None
        if self.save_video:
            os.makedirs(self.debug_dir, exist_ok=True)
            out_path = os.path.join(self.debug_dir, f"debug_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

        exercise_cls = get_exercise_class(self.exercise_name)
        exercise = exercise_cls()
        from src.analytics.workout_summary import WorkoutSummary
        summary = WorkoutSummary()

        metrics = {
            "total_frames": 0,
            "valid_pose_frames": 0,
            "active_frames": 0,
            "vis_sum": 0,
            "current_loss_streak": 0,
            "max_loss_streak": 0,
            "down_transitions": 0,
            "up_transitions": 0
        }

        last_state = "UP"
        start_time = time.time()

        while cap.isOpened():
            success, img = cap.read()
            if not success or img is None:
                break

            metrics["total_frames"] += 1
            frame_idx += 1
            timestamp_ms = int((frame_idx / fps_in) * 1000)

            result = detector.find_pose(img, timestamp_ms)

            pose_found = False
            is_active = False
            feedback = []
            warning = None
            curr_state = last_state

            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                # Updated to handle 6th return: rep_done
                reps, curr_state, is_active, feedback, warning, rep_done = exercise.process(
                    landmarks, world_landmarks, width, height, timestamp_ms, img=img
                )
                
                # Visibility metrics
                exercise_metrics = exercise.get_metrics()
                side = exercise_metrics.get("side")
                
                # Use exercise-specific key indices for visibility tracking
                target_indices = []
                if hasattr(exercise, "L_INDICES") and hasattr(exercise, "R_INDICES"):
                    target_indices = exercise.L_INDICES if side == "L" else exercise.R_INDICES
                
                if target_indices:
                    v_scores = [landmarks[i].visibility for i in target_indices]
                    avg_v = sum(v_scores) / len(v_scores) if v_scores else 0
                else:
                    avg_v = 0
                
                metrics["vis_sum"] += avg_v

                if not warning or "REDUCED" in warning:
                    pose_found = True
                    metrics["valid_pose_frames"] += 1
                    metrics["current_loss_streak"] = 0

                if is_active:
                    metrics["active_frames"] += 1

                if curr_state == "DOWN" and last_state == "UP":
                    metrics["down_transitions"] += 1
                if curr_state == "UP" and last_state == "DOWN":
                    metrics["up_transitions"] += 1

                last_state = curr_state

                if rep_done:
                    summary.add_rep(exercise_metrics["last_score"], {}, feedback, exercise_metrics["angle"])

            if not pose_found:
                metrics["current_loss_streak"] += 1
                metrics["max_loss_streak"] = max(metrics["max_loss_streak"], metrics["current_loss_streak"])
                if not result or not result.pose_landmarks:
                    if hasattr(exercise, 'counter') and exercise.counter:
                        exercise.counter.update(0, "LOW")

            # Debug frame
            debug_img = img.copy()
            
            ex_metrics = exercise.get_metrics()
            last_score = ex_metrics.get("last_score")
            side = ex_metrics.get("side")
            
            # FPS calculation for benchmark (video-time based)
            # Actually, we can just pass fps_in or 0
            
            draw_hud(
                debug_img, 
                exercise.reps, 
                exercise.state, 
                fps_in, 
                feedback, 
                is_active, 
                last_score, 
                warning, 
                side,
                exercise_name=self.exercise_name
            )

            detector.draw_landmarks(debug_img, result)
            debug_img = np.ascontiguousarray(debug_img)

            if debug_img.shape[1] != width or debug_img.shape[0] != height:
                debug_img = cv2.resize(debug_img, (width, height))

            if self.live:
                cv2.imshow("MotionCV Debug", debug_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if out:
                out.write(debug_img)

        cap.release()
        if out: out.release()
        
        # Display final report if live mode
        if self.live:
            final_stats = summary.get_summary()
            if final_stats:
                # Capture last frame for background
                # (Assuming cap was released, we might need a stored last frame)
                # But since we're in the same function, we can just use the last 'debug_img'
                report_img = debug_img.copy()
                draw_session_report(report_img, final_stats)
                cv2.imshow("MotionCV Debug", report_img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

        duration = time.time() - start_time
        avg_fps = metrics["total_frames"] / duration if duration > 0 else 0
        expected = self.ground_truth.get(filename, 0)

        return {
            "video_name": filename,
            "expected_reps": expected,
            "detected_reps": exercise.reps,
            "error": exercise.reps - expected if expected > 0 else 0,
            "avg_fps": round(avg_fps, 2),
            "total_frames": metrics["total_frames"],
            "pose_coverage_pct": metrics["valid_pose_frames"] / metrics["total_frames"] * 100 if metrics["total_frames"] > 0 else 0,
            "avg_visibility": metrics["vis_sum"] / metrics["total_frames"] if metrics["total_frames"] > 0 else 0,
            "max_pose_loss_streak": metrics["max_loss_streak"],
            "active_frames": metrics["active_frames"],
            "active_coverage_pct": metrics["active_frames"] / metrics["total_frames"] * 100 if metrics["total_frames"] > 0 else 0,
            "down_transitions": metrics["down_transitions"],
            "up_transitions": metrics["up_transitions"],
            "aborted_squats": metrics["down_transitions"] - metrics["up_transitions"]
        }

    def save_results(self, results):
        os.makedirs("benchmark", exist_ok=True)

        with open(self.results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved to {self.results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex", type=str, default="squats", choices=["squats", "lunges", "split_squats", "pushups", "curls"],
                        help="Select exercise to benchmark (default: squats)")
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--live", action="store_true")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        exercise=args.ex,
        save_video=args.save_debug_video,
        live=args.live
    )

    runner.run_benchmark()