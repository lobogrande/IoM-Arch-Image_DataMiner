import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 7
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_subtractive_tester():
    # 1. Load Background Templates
    bg_templates = []
    t_path = "templates"
    for f in os.listdir(t_path):
        if f.startswith("background"):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None:
                bg_templates.append(cv2.resize(img, (48, 48)))

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    raw_img = cv2.imread(os.path.join(run_path, files[0]))
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    print(f"--- Subtractive Difference Test: Floor {TARGET_FLOOR} ---")

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        roi = gray[cy-24:cy+24, cx-24:cx+24]
        
        # Find the background template that fits best
        best_diff_score = 999999
        
        for bg in bg_templates:
            # Absolute difference shows what 'doesn't belong' in the slot
            diff = cv2.absdiff(roi, bg)
            score = np.sum(diff) # Total pixel intensity of the difference
            if score < best_diff_score:
                best_diff_score = score

        # Normalize score relative to pixel count
        norm_diff = best_diff_score / (48*48)
        
        # Color Logic: 
        # Low diff (< 15) = High probability of background
        # High diff (> 25) = High probability of occupancy
        color = (0, 255, 0) if norm_diff > 25 else (255, 0, 0)
        
        cv2.rectangle(raw_img, (cx-24, cy-24), (cx+24, cy+24), color, 1)
        cv2.putText(raw_img, f"D:{int(norm_diff)}", (cx-20, cy+20), 0, 0.35, color, 1)

    cv2.imwrite(f"Subtractive_Test_F{TARGET_FLOOR}.jpg", raw_img)
    print(f"Saved Subtractive_Test_F{TARGET_FLOOR}.jpg")
    print("Check if 'D' scores effectively separate blocks from floor noise.")

if __name__ == "__main__":
    run_subtractive_tester()