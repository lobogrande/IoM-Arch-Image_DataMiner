import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 18
PROBLEM_SLOTS = [2, 7, 16, 22] # The slots you identified
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
UNIFIED_ROOT = "Unified_Consensus_Inputs"

def run_truth_finder():
    # 1. Load ALL Ore Templates into a list
    all_templates = []
    t_path = "templates"
    for f in os.listdir(t_path):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is not None:
            all_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    # 2. Load the Floor Image
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    # (Assuming the sequence lookup is handled or file is accessible)
    # For this diagnostic, we'll just assume the file name is known or find it
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    img = cv2.imread(os.path.join(run_path, files[0]), 0)

    print(f"--- TRUTH FINDER: SLOT LEADERBOARD ---")

    for slot in PROBLEM_SLOTS:
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        roi = img[cy-24:cy+24, cx-24:cx+24]
        
        # Test against every template
        matches = []
        for t in all_templates:
            score = cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED).max()
            matches.append((t['name'], score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)

        print(f"\n[SLOT {slot}] Top 3 Matches:")
        for i in range(3):
            name, score = matches[i]
            print(f"  {i+1}. {score:.4f} -> {name}")
            
        if matches[0][1] < 0.80:
            print(f"  [!] ALERT: Best match is BELOW our 0.80 strict threshold.")

if __name__ == "__main__":
    run_truth_finder()