# Lunge Benchmark Plan

Objective: Empirically validate the accuracy and robustness of the Generalized Lunge Engine across all supported variations and environmental conditions.

## 1. Test Categories & Expected Behavior

| Category | Description | Primary Metric | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **Stationary (Split Squat)** | Vertical movement, feet fixed. | Rep Count | Lead leg remains locked; high accuracy (>95%). |
| **Reverse Lunge** | Step back, return to neutral. | Rep Count | Working leg identified at descent; reset at neutral. |
| **Alternating Lunge** | Step forward/back, swap legs. | Side Accuracy | `working_side` must flip (L->R) every rep. |
| **Walking Lunge** | Step forward continuously. | Baseline Stability | Hip baseline recalibrates correctly after each stride. |
| **Bulgarian Split Squat** | Rear foot elevated on bench. | False-Pos Reject | Rear leg never selected as working leg; depth accurate. |

## 2. Failure Criteria

A test video is considered a **FAILURE** if:
1. **Rep Count Error**: Detected reps != Expected reps (Ground Truth).
2. **Identity Drift**: Tracker jumps to a background person for >10 frames.
3. **Side Hallucination**: Rear leg in Bulgarian Split Squat is counted as a rep.
4. **Initialization Stall**: HUD stays in "INITIALIZING..." while athlete is in frame.
5. **Double Count**: A single lunge triggers 2 rep counts due to jitter or reset failure.

## 3. Edge Cases to Validate

- **Weighted (Dumbbell)**: Weights held at sides partially obscuring knees/hips.
- **Occluded Background**: People walking behind the athlete.
- **Partial View**: Camera cropped to waist-down only.
- **High Tempo**: Rapid alternating lunges with <0.5s neutral phase.
- **Depth Variance**: Very shallow reps (should be rejected) vs. deep reps.

## 4. Benchmark Results Schema (`lunge_benchmark.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `video_id` | String | Filename or unique ID. |
| `category` | Enum | Stationary, Reverse, Alternating, Walking, Bulgarian. |
| `expected_reps` | Int | Manual ground truth count. |
| `detected_reps` | Int | Count reported by the engine. |
| `side_errors` | Int | Number of times the wrong leg was identified. |
| `false_positives` | Int | Reps counted during jitters or non-movement. |
| `identity_lost` | Boolean | True if focus switched to another person. |
| `max_bone_drift_pct` | Float | Maximum femur/tibia length variance observed. |
| `notes` | String | Specific failure mode observed (e.g., "Rear leg flex"). |

## 5. Execution Strategy

1. **Baseline**: Run 5 stationary lunge videos to confirm basic kinematic triggers.
2. **Stress Test**: Run 3 Bulgarian and 3 Walking lunge videos to test reset/recalibration logic.
3. **Gym Noise**: Run 2 videos with background movement to validate Identity Lock.
4. **Occlusion**: Run 2 videos with heavy dumbbells to validate Bone Rejection.
