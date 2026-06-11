class WorkoutSummary:
    def __init__(self):
        self.reps_data = []

    def add_rep(self, score, breakdown, feedback, min_angle):
        self.reps_data.append({
            "score": score,
            "breakdown": breakdown,
            "feedback": feedback,
            "min_angle": min_angle
        })

    def get_summary(self):
        if not self.reps_data:
            return None

        total_reps = len(self.reps_data)
        # Use 0 as default for None scores
        avg_score = sum((r["score"] if r["score"] is not None else 0) for r in self.reps_data) / total_reps
        best_rep = max(self.reps_data, key=lambda x: (x["score"] if x["score"] is not None else 0))
        
        # Aggregate feedback (count occurrences of specific cues)
        # We only want to count "constructive" cues (the orange ones)
        feedback_counts = {}
        for r in self.reps_data:
            for cue in r["feedback"]:
                # Ignore the positive reinforcement for the summary
                if any(word in cue.lower() for word in ["excellent", "great", "perfect", "good", "target", "stable"]):
                    continue
                feedback_counts[cue] = feedback_counts.get(cue, 0) + 1
        
        # Sort feedback by frequency
        sorted_feedback = sorted(feedback_counts.items(), key=lambda x: x[1], reverse=True)
        top_cues = [f[0] for f in sorted_feedback[:3]]

        return {
            "total_reps": total_reps,
            "average_score": int(avg_score),
            "best_score": best_rep["score"],
            "top_cues": top_cues,
            "reps": self.reps_data
        }
