import cv2
import numpy as np
import os

# --- FORENSIC RANGE ---
DIAG_START = 2000
DIAG_END = 2200

# --- PRODUCTION CONSTANTS ---
HEADER_ROI = (54, 74, 103, 138)
PLAYER_ROI_Y = (120, 420)

def run_v5_08_forensic_hud_trace():
    buffer_root = "capture_buffer_0"
    p_right = cv2.imread("templates/player_right.png", 0)
    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # Baseline for diff
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor_hud = f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]

    print(f"--- Running v5.08: Forensic HUD Trace ---")
    print(f"Index | HUD Diff | Player Score | Action")

    for i in range(DIAG_START, min(DIAG_END, len(files))):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        
        # Calculate HUD Shift
        diff = np.mean(cv2.absdiff(cur_hud, anchor_hud))
        
        # Calculate Player Presence
        roi = img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :]
        p_score = cv2.minMaxLoc(cv2.matchTemplate(roi, p_right, cv2.TM_CCOEFF_NORMED))[1]
        
        action = "!!!" if diff > 3.8 else "."
        print(f"{i:05} | {diff:8.3f} | {p_score:12.3f} | {action}")
        
        # Update anchor if a 'hit' occurred (simulating the logic)
        if diff > 3.8:
            anchor_hud = cur_hud.copy()

if __name__ == "__main__":
    run_v5_08_forensic_hud_trace()