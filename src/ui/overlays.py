import cv2

def draw_hud(img, reps, state, fps, feedback_list, is_active, last_score=None, warning_msg=None, side=None, exercise_name="SQUAT"):
    h, w, _ = img.shape
    
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
    for i, feedback in enumerate(feedback_list):
        # Green for positive reinforcement, Orange for cues/warnings
        positive_keywords = ["excellent", "great", "target", "neutral", "stable", "upright", "depth", "perfect", "good"]
        fb_color = (0, 255, 0) if any(word in feedback.lower() for word in positive_keywords) else (0, 165, 255)
        cv2.putText(img, f"- {feedback}", (25, 280 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)
    
    if side:
        cv2.putText(img, f"SIDE: {side}", (w - 100, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    if fps > 0:
        cv2.putText(img, f"FPS: {int(fps)}", (w - 100, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
