from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import uvicorn
import time

# Import our perfected logic
from src.exercises import get_exercise_class
from src.pose.pose_detector import PoseDetector, TrackingState

app = FastAPI()

# In-memory storage for session state
# Stores {"session_id": {"exercise": CurlObj, "detector": PoseDetector}}
session_data = {}

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float

class FrameData(BaseModel):
    session_id: str
    exercise: str
    landmarks: List[Landmark]
    world_landmarks: Optional[List[Landmark]] = None
    width: int
    height: int
    timestamp_ms: int

@app.get("/")
def home():
    return {"status": "MotionCV Engine Online", "active_sessions": len(session_data)}

@app.post("/process")
async def process_frame(data: FrameData):
    # 1. Get or create session (Each user needs their own FSM and Counter)
    if data.session_id not in session_data:
        ex_cls = get_exercise_class(data.exercise)
        session_data[data.session_id] = {
            "exercise": ex_cls(),
            "detector": PoseDetector() 
        }
    
    sess = session_data[data.session_id]
    exercise = sess["exercise"]
    detector = sess["detector"]
    
    # 2. Convert landmarks to Internal Objects
    class SimpleLM:
        def __init__(self, d):
            self.x, self.y, self.z, self.visibility = d.x, d.y, d.z, d.visibility
            
    raw_lms = [SimpleLM(lm) for lm in data.landmarks]
    
    # Mock result for FSM logic
    class MockResult:
        def __init__(self, lms):
            self.pose_landmarks = [lms]
            self.pose_world_landmarks = None
            
    # 3. Apply FSM Tracking (The "Bed Fix" logic)
    # We manually update the detector state based on the landmarks sent by Flutter
    # Since Flutter already did the detection, we use the FSM for 'Lock' and 'Centroid' consistency
    mock_result = MockResult(raw_lms)
    
    # Check if we are currently mid-rep to harden the tracker
    is_active_rep = getattr(exercise, 'is_active', False)
    
    # We reuse the update logic from find_pose but with the landmarks already provided
    best_pose = raw_lms
    curr_centroid = detector._get_centroid(best_pose)
    
    tracking_valid = True
    if detector.last_centroid is not None and curr_centroid is not None:
        dist = np.linalg.norm(curr_centroid - detector.last_centroid)
        jump_limit = 0.45 if not is_active_rep else 0.70
        if dist > jump_limit:
            detector.jump_frames += 1
            if detector.jump_frames < 4:
                tracking_valid = False # Ignore the jump
    
    if tracking_valid:
        detector._update_state(best_pose, 1.0) # High confidence as it's pre-detected
    else:
        detector._handle_no_detection()

    # 4. Run Exercise Logic ONLY if we are LOCKED or REACQUIRING
    # This prevents the Bed from counting reps
    if detector.state in [TrackingState.LOCKED, TrackingState.REACQUIRING]:
        reps, state, active, feedback, warning, rep_done = exercise.process(
            raw_lms, 
            None, 
            data.width, 
            data.height, 
            data.timestamp_ms
        )
    else:
        # Tracker is LOST or SEARCHING - don't process movement
        reps = exercise.reps
        state = "SEARCHING"
        active = False
        feedback = ["Align your body in frame"]
        rep_done = False

    # 5. Return Results
    metrics = exercise.get_metrics()
    
    return {
        "reps": reps,
        "state": state,
        "is_active": active,
        "feedback": feedback,
        "last_score": metrics.get("last_score"),
        "active_side": metrics.get("side"),
        "rep_done": rep_done,
        "tracking_state": detector.state,
        "key_joints": getattr(exercise, "KEY_JOINTS", [])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
