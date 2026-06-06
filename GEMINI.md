# GEMINI.md

# MotionCV

## Mission

MotionCV is an AI-powered fitness coach that uses computer vision to provide real-time exercise feedback, rep counting, form analysis, and long-term movement improvement.

Computer vision is not the product.

Coaching is the product.

The objective is not to detect poses. The objective is to help users move better.

---

# Current Milestone (ACTIVE)

This is the only milestone that matters right now.

## Goal

Build a reliable squat coaching system.

A user should be able to:

1. Stand in front of a camera
2. Perform squats
3. Receive accurate rep counting
4. Receive useful real-time feedback

## Definition of Done

A user can perform 100 consecutive squats and receive:

* Accurate rep counting
* Stable pose tracking
* Real-time corrections
* Per-rep scores
* Consistent performance

Nothing should delay this milestone.

Do not implement future features until this milestone is complete.

---

# Product Vision

The long-term goal is to create an experience that feels like training with a coach.

A user should be able to:

1. Open the app
2. Start exercising immediately
3. Receive live corrections
4. Review mistakes after each set
5. Track progress over time

The experience should feel intelligent, personalized, and actionable.

---

# Product Principles

Every feature should answer:

"How does this help the user improve?"

Prefer:

* Actionable feedback
* Simplicity
* Reliability
* Explainability
* Fast iteration

Avoid complexity unless it directly improves user outcomes.

---

# Development Philosophy

Prefer:

* Simple solutions
* Deterministic systems
* Explainable logic
* Real-time performance

Do not introduce machine learning solely because it is available.

Introduce ML only when it provides measurable user value.

---

# Technical Stack

## Current

* Python
* OpenCV
* MediaPipe Pose
* NumPy

## Future

Only introduce if necessary:

* PyTorch
* TensorFlow Lite
* ONNX Runtime

---

# Architecture

Camera / Video
→ Pose Detection
→ Landmark Smoothing
→ Feature Extraction
→ Exercise Engine
→ Rep Counter
→ Form Analyzer
→ Coaching Engine
→ Analytics
→ User Interface

---

# Project Structure

MotionCV/

src/

pose/

* pose_detector.py
* landmark_smoother.py

core/

* angle_utils.py
* geometry_utils.py
* state_machine.py

analysis/

* rep_counter.py
* form_analyzer.py
* scoring_engine.py
* coaching_engine.py

exercises/

* squat.py
* pushup.py
* lunge.py
* curl.py
* pullup.py

analytics/

* workout_summary.py
* symmetry_analysis.py

ui/

* overlays.py

videos/

* test_videos/

docs/

main.py

---

# Development Order

Implement features in this exact order.

## Phase 1

* Pose Detection
* Landmark Visualization
* Landmark Smoothing

## Phase 2

* Angle Calculation
* Squat State Machine
* Rep Counter

## Phase 3

* Squat Form Analysis
* Real-Time Feedback

## Phase 4

* Per-Rep Scoring
* Workout Summary

## Phase 5

* Push-Up Support
* Curl Support
* Lunge Support
* Pull-Up Support

## Phase 6

* Exercise Recognition
* Symmetry Analysis
* Movement Replay

---

# Feature Priority

## Priority 1

Must exist before anything else.

* Pose Tracking
* Angle Extraction
* Rep Counting
* Form Analysis
* Real-Time Coaching

## Priority 2

Creates meaningful user value.

* Per-Rep Scoring
* Workout Summaries
* Progress Tracking
* Form Breakdown Reports

## Priority 3

Differentiators.

* Exercise Recognition
* Symmetry Analysis
* Replay System
* Personal Baselines
* Fatigue Detection

## Priority 4

Advanced AI Features.

* ML Form Evaluation
* Personalized Coaching Models
* Exercise Recommendation Systems

---

# Supported Exercises

Initial exercises:

1. Squat
2. Push-Up
3. Bicep Curl
4. Lunge
5. Pull-Up

Every exercise should reuse shared infrastructure.

Avoid duplicated logic.

---

# Rep Counting Rules

Rep counting should be deterministic whenever possible.

Preferred:

* Joint Angles
* State Machines
* Range of Motion Tracking

Avoid:

* Frame Counting
* Timing Heuristics
* Magic Numbers

Every counted rep should be explainable.

---

# Form Analysis Rules

Feedback must be:

* Immediate
* Specific
* Actionable
* Explainable

Good examples:

* Go deeper
* Keep chest up
* Push knees outward
* Fully extend elbows
* Slow down

Bad examples:

* Poor form
* Incorrect movement
* Low score

Always explain the reason.

---

# Scoring Philosophy

Scores are secondary.

Coaching is primary.

Never show a score without an explanation.

Example:

Rep 8

Depth: 28/30
Posture: 23/25
Stability: 19/25
Tempo: 16/20

Total: 86/100

Needs Improvement:

* Slight knee collapse
* Inconsistent depth

---

# Real-Time Coaching

MotionCV should behave like a coach.

Examples:

* Go deeper
* Chest up
* Maintain balance
* Slow down
* Good rep
* Excellent depth

Feedback should occur during movement, not only after the set.

---

# Performance Requirements

Target:

* 20+ FPS
* Low latency
* Stable landmark tracking
* Consumer hardware compatibility

Responsiveness is more important than perfect accuracy.

---

# Testing Requirements

Every feature must be tested using:

* Webcam input
* Recorded videos
* Multiple camera angles
* Different lighting conditions
* Different body types

Never validate a feature using only one test video.

---

# Explicit Non-Goals

Do NOT:

* Train custom pose models
* Fine-tune pose estimation models
* Replace MediaPipe without evidence
* Introduce YOLO for pose tracking
* Build cloud infrastructure
* Collect large datasets before proving value
* Implement advanced ML before Phase 6

---

# Decision Framework

Before implementing anything, ask:

1. Does this improve coaching quality?
2. Does this improve user experience?
3. Does this improve reliability?
4. Is it explainable?
5. Is there a simpler solution?

If not, do not build it.

---

# North Star

The goal is not to count repetitions.

The goal is to help users move better.

Every technical decision should support that objective.
