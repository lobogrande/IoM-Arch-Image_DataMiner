import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json

# --- 1. TARGET CONFIG ---
TEST_FLOORS = [11, 18] # Boss vs. Noisy
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# The Matrix: (Threshold, Delta)
# We want to find the one that keeps F11 full and F18 empty.
SETTINGS = [
    (0.77, 0.04), # Set A: Moderate
    (0.78, 0.05), # Set B: Firm
    (0.80, 0.06), # Set C: Strict (Your successful test)
    (0.81, 0.07)  # Set D: Ultra-Strict
]

def run_calibration_matrix():
    # Load Templates
    templates = {'ore': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"):
            templates['bg'].append(img)
        else:
            p = f.split("_")
            if p[0] not in templates['ore']: templates['ore'][p[0]] = {'act': [], 'sha': []}
            templates['ore'][p[0]][p[1]].append(img)

    run_path = os.path.join(UNIFIED_ROOT, "Run_0")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    print("--- Running Calibration Matrix: A, B, C, D ---")

    for f_num in TEST_FLOORS:
        base_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{seq[f_num]['frame']}"))
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        
        for name, (thresh, delta_val) in zip(['A','B','C','D'], SETTINGS):
            canvas = base_img.copy()
            for slot in range(24):
                row, col = divmod(slot, 6)
                cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
                x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
                roi = gray[y1:y2, x1:x2]

                bg_score = max([cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max() for t in templates['bg']])
                
                best_ore = {'score': 0.0, 'tier': ''}
                for tier, states in templates['ore'].items():
                    for state, imgs in states.items():
                        for t_img in imgs:
                            s = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
                            if state == 'sha': s *= 1.03
                            if s > best_ore['score']:
                                best_ore = {'score': s, 'tier': tier}

                # Competitive Logic
                if best_ore['score'] > thresh and (best_ore['score'] - bg_score) > delta_val:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

            overlay_text = f"Set {name}: Thresh {thresh} / Delta {delta_val}"
            cv2.putText(canvas, overlay_text, (20, 40), 0, 0.6, (0, 255, 255), 2)
            cv2.imwrite(f"Calibration_F{f_num}_{name}.jpg", canvas)
            print(f" [+] Generated F{f_num} {name}")

if __name__ == "__main__":
    run_calibration_matrix()