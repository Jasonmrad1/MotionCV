import csv
import os

def generate_summary():
    results_file = "benchmark/benchmark_results.csv"
    if not os.path.exists(results_file):
        print("Results file not found. Run benchmark_runner.py first.")
        return

    total_videos = 0
    total_fps = 0
    total_pose_cov = 0
    total_active_cov = 0
    total_reps = 0
    
    # Rep counting accuracy metrics
    valid_expected_count = 0
    total_abs_error = 0

    print("\n" + "="*80)
    print(f"{'VIDEO NAME':<25} | {'REPS (D/E)':<12} | {'FPS':<8} | {'POSE COV':<10} | {'ACTIVE':<8}")
    print("-" * 80)

    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_videos += 1
            name = row["video_name"]
            detected = int(row["detected_reps"])
            expected = int(row["expected_reps"]) if row["expected_reps"] != "0" else 0
            fps = float(row["avg_fps"])
            pose_cov = float(row["pose_coverage_pct"])
            active_cov = float(row["active_coverage_pct"])
            
            total_fps += fps
            total_pose_cov += pose_cov
            total_active_cov += active_cov
            total_reps += detected
            
            reps_str = f"{detected}/{expected}" if expected > 0 else f"{detected}/--"
            
            if expected > 0:
                valid_expected_count += 1
                total_abs_error += abs(detected - expected)

            print(f"{name[:25]:<25} | {reps_str:<12} | {fps:<8.1f} | {pose_cov:<9.1f}% | {active_cov:<7.1f}%")

    print("="*80)
    print(f"SUMMARY ({total_videos} videos)")
    print(f"  Average FPS: {total_fps / total_videos:.1f}")
    print(f"  Avg Pose Coverage: {total_pose_cov / total_videos:.1f}%")
    print(f"  Avg Active Coverage: {total_active_cov / total_videos:.1f}%")
    print(f"  Total Reps Detected: {total_reps}")
    
    if valid_expected_count > 0:
        mae = total_abs_error / valid_expected_count
        print(f"  Rep Count Mean Absolute Error: {mae:.2f}")
    else:
        print("  (Annotate 'expected_reps' in benchmark_results.csv to see accuracy metrics)")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_summary()
