import cv2
import time
import os
import csv
import json
import argparse
import numpy as np

from src.pose.pose_detector import PoseDetector
from src.pose.landmark_smoother import LandmarkSmoother
from src.analysis.angle_utils import compute_angle
from src.analysis.rep_counter import SquatCounter
from src.analysis.form_analyzer import FormAnalyzer
from src.analysis.scoring_engine import ScoringEngine


class BenchmarkRunner:
    def __init__(self, exercise="squats", save_video=False, live=False):
        self.exercise = exercise
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
                        if row.get("expected_reps"):
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

        # =========================================================
        # FIX 1: reliable width/height (IMPORTANT)
        # =========================================================
        ret, test_frame = cap.read()
        if not ret:
            print(f"[ERROR] Cannot read video: {filename}")
            return None

        height, width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # =========================================================

        frame_idx = 0

        # Debug video writer
        out = None
        if self.save_video:
            os.makedirs(self.debug_dir, exist_ok=True)
            out_path = os.path.join(self.debug_dir, f"debug_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # FIX: correct size guaranteed
            out = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

        counter = SquatCounter(min_visibility=0.15, buffer_limit=30)
        analyzer = FormAnalyzer()
        scorer = ScoringEngine()
        smoother = LandmarkSmoother(alpha=0.6)

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

        event_logs = []
        last_state = "UP"
        start_time = time.time()

        L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
        L_INDICES = [L_HIP, L_KNEE, L_ANKLE]
        R_INDICES = [24, 26, 28]
        KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]
        last_valid_coords = {i: None for i in KEY_JOINTS}

        while cap.isOpened():
            success, img = cap.read()
            if not success or img is None:
                break

            metrics["total_frames"] += 1
            frame_idx += 1
            timestamp_ms = int((frame_idx / fps_in) * 1000)

            result = detector.find_pose(img, timestamp_ms)

            pose_found = False
            angle = None
            is_active = False
            conf_level = "LOW"
            curr_state = last_state

            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                side_prefix, conf_level = counter.get_best_side_confidence(
                    landmarks, L_INDICES, R_INDICES
                )

                v_scores = [
                    landmarks[i].visibility
                    for i in (L_INDICES if side_prefix == "L" else R_INDICES)
                ]

                avg_v = sum(v_scores) / 3
                metrics["vis_sum"] += avg_v

                if conf_level != "LOW":
                    pose_found = True
                    metrics["valid_pose_frames"] += 1
                    metrics["current_loss_streak"] = 0

                    HIP = 23 if side_prefix == "L" else 24
                    KNEE = 25 if side_prefix == "L" else 26
                    ANKLE = 27 if side_prefix == "L" else 28
                    SHOULDER = 11 if side_prefix == "L" else 12

                    def get_target_pt(idx, is_leg=True):
                        current_v = landmarks[idx].visibility
                        if current_v >= 0.4:
                            coords = (landmarks[idx].x * width,
                                      landmarks[idx].y * height)
                            last_valid_coords[idx] = coords
                            return coords
                        if is_leg and current_v < 0.15:
                            return None
                        return last_valid_coords[idx]

                    hip = smoother.smooth(HIP, get_target_pt(HIP, is_leg=True))
                    knee = smoother.smooth(KNEE, get_target_pt(KNEE, is_leg=True))
                    ankle = smoother.smooth(ANKLE, get_target_pt(ANKLE, is_leg=True))
                    shoulder = get_target_pt(SHOULDER, is_leg=False)

                    if hip and knee and ankle:
                        angle = compute_angle(hip, knee, ankle)
                        _, curr_state, is_viable, rep_done, is_active, duration = counter.update(
                            angle, conf_level, timestamp_ms / 1000.0
                        )

                        if is_viable:
                            if is_active and shoulder:
                                analyzer.analyze_posture(hip, shoulder, knee, ankle, angle, is_active)

                            if counter.last_rep_data:
                                rd = counter.last_rep_data
                                status = "COUNTED" if rd["valid"] else "REJECTED"
                                print(f"  [REP {status}] Frame {frame_idx} | {rd['reason']} | ROM: {rd['rom']} | Duration: {rd['duration']}s | Min Angle: {rd['min_angle']}")

                            if rep_done:
                                feedback, min_angle, max_tilt = analyzer.get_rep_summary()
                                scorer.calculate_score(min_angle, max_tilt, duration)
                                analyzer.reset_rep_metrics()

                        if is_active:
                            metrics["active_frames"] += 1

                        if curr_state == "DOWN" and last_state == "UP":
                            metrics["down_transitions"] += 1
                        if curr_state == "UP" and last_state == "DOWN":
                            metrics["up_transitions"] += 1

                        last_state = curr_state

            if not pose_found:
                metrics["current_loss_streak"] += 1
                metrics["max_loss_streak"] = max(
                    metrics["max_loss_streak"],
                    metrics["current_loss_streak"]
                )
                counter.update(0, "LOW")

            event_logs.append({
                "ts": timestamp_ms,
                "angle": round(angle, 1) if angle else None,
                "state": counter.state,
                "active": is_active,
                "reps": counter.count
            })

            # =========================
            # DEBUG FRAME
            # =========================
            debug_img = img.copy()

            cv2.rectangle(debug_img, (10, 10), (420, 280), (0, 0, 0), -1)

            cv2.putText(debug_img, f"REPS: {counter.count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(debug_img, f"STATE: {counter.state}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            cv2.putText(debug_img, f"ANGLE: {int(angle) if angle else 'N/A'}",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(debug_img, f"ACTIVE: {is_active}", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.putText(debug_img,
                        f"DOWN:{metrics['down_transitions']} UP:{metrics['up_transitions']}",
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            detector.draw_landmarks(debug_img, result)

            # =========================================================
            # FIX 2: frame safety before writing
            # =========================================================
            debug_img = np.ascontiguousarray(debug_img)

            if debug_img.shape[1] != width or debug_img.shape[0] != height:
                debug_img = cv2.resize(debug_img, (width, height))
            # =========================================================

            if self.live:
                cv2.imshow("MotionCV Debug", debug_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if out:
                out.write(debug_img)

        cap.release()
        if out:
            out.release()

        cv2.destroyAllWindows()

        duration = time.time() - start_time
        avg_fps = metrics["total_frames"] / duration if duration > 0 else 0

        expected = self.ground_truth.get(filename, 0)

        return {
            "video_name": filename,
            "expected_reps": expected,
            "detected_reps": counter.count,
            "error": counter.count - expected if expected > 0 else 0,
            "avg_fps": round(avg_fps, 2),
            "total_frames": metrics["total_frames"],
            "pose_coverage_pct":
                metrics["valid_pose_frames"] / metrics["total_frames"] * 100
                if metrics["total_frames"] > 0 else 0,
            "avg_visibility":
                metrics["vis_sum"] / metrics["total_frames"]
                if metrics["total_frames"] > 0 else 0,
            "max_pose_loss_streak": metrics["max_loss_streak"],
            "active_frames": metrics["active_frames"],
            "active_coverage_pct":
                metrics["active_frames"] / metrics["total_frames"] * 100
                if metrics["total_frames"] > 0 else 0,
            "down_transitions": metrics["down_transitions"],
            "up_transitions": metrics["up_transitions"],
            "aborted_squats":
                metrics["down_transitions"] - metrics["up_transitions"]
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

    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--live", action="store_true")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        save_video=args.save_debug_video,
        live=args.live
    )

    runner.run_benchmark()