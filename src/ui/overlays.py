import cv2
import time

# Global state for feedback persistence
feedback_start_time = 0
current_persisted_feedback = []

def draw_hud(img, reps, state, fps, feedback_list, is_active, last_score=None, warning_msg=None, side=None, exercise_name="SQUAT"):
    global feedback_start_time, current_persisted_feedback
    h, w, _ = img.shape
    curr_time = time.time()
    
    # 1. Feedback Persistence Logic (Min 1.5s hold)
    if feedback_list:
        # If new actionable feedback arrived, or if current is empty
        if not current_persisted_feedback or feedback_list != current_persisted_feedback:
            # Only update if current persistence window has expired
            if curr_time - feedback_start_time > 1.5:
                current_persisted_feedback = feedback_list
                feedback_start_time = curr_time
    
    # Clear feedback if it's old AND no new feedback is provided
    if not feedback_list and (curr_time - feedback_start_time > 1.5):
        current_persisted_feedback = []

    display_feedback = current_persisted_feedback if current_persisted_feedback else feedback_list

    # Header bar
    if warning_msg:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, warning_msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(img, "Setup: Side View | Hip Height | 6ft Distance", (w - 450, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Sidebar
    panel_w = 300
    panel_h = 320
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 60), (panel_w, 60 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Status colors
    active_color = (0, 255, 0) if is_active else (150, 150, 150)
    rep_color = (0, 255, 0) if not warning_msg else (0, 0, 255)
    
    cv2.putText(img, f"REPS: {reps}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rep_color, 3)
    cv2.putText(img, f"STATE: {state}", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2)
    
    if last_score is not None:
        score_color = (0, 255, 0) if last_score > 80 else (0, 165, 255) if last_score > 60 else (0, 0, 255)
        cv2.putText(img, f"SCORE: {last_score}", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)

    cv2.putText(img, f"{exercise_name.upper()} ACTIVE" if is_active else "IDLE", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1)
    
    cv2.putText(img, "LATEST FEEDBACK:", (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for i, feedback in enumerate(display_feedback):
        # STRICT COLOR CODING
        # Green = Positive Reinforcement / Achievement
        # Orange = Actionable Cue / Correction
        # Red = Safety Warning
        positive_keywords = ["excellent", "great", "perfect", "good", "target", "stable", "upright", "strong", "explosive", "depth", "safe", "solid", "elite"]
        warning_keywords = ["tuck", "sag", "tilt", "safety", "impingement"]
        
        if any(word in feedback.lower() for word in warning_keywords):
            fb_color = (0, 0, 255) # Red
        elif any(word in feedback.lower() for word in positive_keywords):
            fb_color = (0, 255, 0) # Green
        else:
            fb_color = (0, 165, 255) # Orange
            
        cv2.putText(img, f"- {feedback}", (25, 280 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)
    
    if side:
        cv2.putText(img, f"SIDE: {side}", (w - 100, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    if fps > 0:
        cv2.putText(img, f"FPS: {int(fps)}", (w - 100, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def draw_session_report(img, final_stats):
    """
    Draws a full-screen semi-transparent overlay with workout statistics.
    """
    h, w, _ = img.shape
    
    # 1. Background Blur/Darken
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # 2. Card Background
    card_w, card_h = 600, 450
    cx, cy = w // 2, h // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)

    # 3. Header
    cv2.putText(img, "WORKOUT COMPLETE", (x1 + 150, y1 + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # 4. Main Stats
    y_offset = y1 + 130
    stats = [
        ("Total Reps", str(final_stats['total_reps']), (255, 255, 255)),
        ("Avg Session Score", str(final_stats['average_score']), (0, 255, 0) if final_stats['average_score'] > 80 else (0, 165, 255)),
        ("Best Rep Score", str(final_stats['best_score']), (255, 255, 0))
    ]

    for label, val, color in stats:
        cv2.putText(img, f"{label}:", (x1 + 50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        cv2.putText(img, val, (x1 + 350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y_offset += 50

    # 5. Focus Areas
    cv2.putText(img, "TOP FOCUS AREAS:", (x1 + 50, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y_offset += 70
    
    if final_stats['top_cues']:
        for cue in final_stats['top_cues']:
            cv2.putText(img, f"> {cue}", (x1 + 70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
            y_offset += 40
    else:
        cv2.putText(img, "None! Perfect Technique.", (x1 + 70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    # 6. Footer
    cv2.putText(img, "Press 'ANY KEY' to Close", (x1 + 180, y2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
