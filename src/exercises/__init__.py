from src.exercises.squat import Squat

def get_exercise_class(name):
    exercises = {
        "squats": Squat
    }
    return exercises.get(name.lower(), Squat)
