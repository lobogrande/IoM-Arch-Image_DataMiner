import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 18
PROBLEM_SLOTS = [2, 7, 16, 22]
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def create_ore_mask():
    """Creates a 48x48 mask that ignores the top-right and corners (where icons live)."""
    mask = np.zeros((48, 48), dtype=np.uint8)
    # We focus on a central circle where the core ore pattern is strongest
    cv2.circle(mask, (24, 24), 18, 255, -1) 
    return mask

def run_masked_diagnostic():
    mask = create_ore_mask()
    
    # Load Templates
    all_templates = []
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if f.startswith("background"): continue
        t_img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if t_img is not None:
            all_templates.append({'name': f, 'img': cv2.resize(t_img, (48, 48))})

    # Load Floor
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    img = cv2.imread(os.path.join(run_path, files[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"--- MASKED DIAGNOSTIC: IGNORING ICON CORNERS ---")

    for slot in PROBLEM_SLOTS:
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        roi = gray[cy-24:cy+24, cx-24:cx+24]
        
        matches = []
        for t in all_templates:
            # We use the MASK here to only compare the 'heart' of the ore
            res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            matches.append((t['name'], max_val))
        
        matches.sort(key=lambda x: x[1], reverse=True)

        print(f"\n[SLOT {slot}] Top 3 Masked Matches:")
        for i in range(3):
            print(f"  {i+1}. {matches[i][1]:.4f} -> {matches[i][0]}")

    # Visual Output of the Mask
    cv2.imwrite("Diagnostic_Mask_Preview.png", mask)
    print("\nCheck Diagnostic_Mask_Preview.png to see the 'Zone of Interest'.")

if __name__ == "__main__":
    run_masked_diagnostic()