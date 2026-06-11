from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import uvicorn

# Import our perfected logic
from src.exercises import get_exercise_class
from src.pose.landmark_smoother import LandmarkSmoother

app = FastAPI()

# In-memory storage for session state (for the prototype)
sessions = {}

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
    return {"status": "MotionCV Engine Online"}

@app.post("/process")
async def process_frame(data: FrameData):
    # 1. Get or create session
    if data.session_id not in sessions:
        ex_cls = get_exercise_class(data.exercise)
        sessions[data.session_id] = ex_cls()
    
    exercise = sessions[data.session_id]
    
    # 2. Convert Pydantic landmarks back to the format our engine expects
    # Our engine expects a list of objects with .x, .y, .z, .visibility
    class SimpleLM:
        def __init__(self, d):
            self.x, self.y, self.z, self.visibility = d.x, d.y, d.z, d.visibility
            
    landmarks = [SimpleLM(lm) for lm in data.landmarks]
    
    # Mock world_landmarks object structure for world_landmarks.landmark[idx]
    class WorldLM:
        def __init__(self, lms):
            self.landmark = lms
            
    world_lms = None
    if data.world_landmarks:
        world_lms = WorldLM([SimpleLM(lm) for lm in data.world_landmarks])

    # 3. Run the "Secret Sauce" logic
    reps, state, active, feedback, warning, rep_done = exercise.process(
        landmarks, 
        world_lms, 
        data.width, 
        data.height, 
        data.timestamp_ms
    )
    
    # 4. Get metrics (score, etc)
    metrics = exercise.get_metrics()
    
    return {
        "reps": reps,
        "state": state,
        "is_active": active,
        "feedback": feedback,
        "last_score": metrics.get("last_score"),
        "active_side": metrics.get("side"),
        "rep_done": rep_done
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
