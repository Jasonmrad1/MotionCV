class WorkoutSummary:
    def __init__(self):
        self.reps_data = []

    def add_rep(self, score, breakdown, feedback):
        self.reps_data.append({
            "score": score,
            "breakdown": breakdown,
            "feedback": feedback
        })

    def get_summary(self):
        if not self.reps_data:
            return None

        total_reps = len(self.reps_data)
        avg_score = sum(r["score"] for r in self.reps_data) / total_reps
        
        # Aggregate feedback (count occurrences of specific cues)
        feedback_counts = {}
        for r in self.reps_data:
            for cue in r["feedback"]:
                feedback_counts[cue] = feedback_counts.get(cue, 0) + 1
        
        # Sort feedback by frequency
        sorted_feedback = sorted(feedback_counts.items(), key=lambda x: x[1], reverse=True)
        top_cues = [f[0] for f in sorted_feedback[:3]]

        return {
            "total_reps": total_reps,
            "average_score": int(avg_score),
            "top_cues": top_cues
        }
