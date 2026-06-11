from src.exercises.squat import Squat
from src.exercises.lunge import Lunge
from src.exercises.split_squat import SplitSquat
from src.exercises.pushup import PushUp
from src.exercises.curl import Curl

def get_exercise_class(name):
    exercises = {
        "squats": Squat,
        "lunges": Lunge,
        "split_squats": SplitSquat,
        "pushups": PushUp,
        "curls": Curl
    }
    return exercises.get(name.lower(), Squat)

def get_available_exercises():
    return ["squats", "lunges", "split_squats", "pushups", "curls"]
