import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- CALIBRATED CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 11)
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# THE NEW GATES (Calibrated for Run 0 Noise)
D_GATE = 6      # Lowered from 18 to catch 'quiet' ores
O_GATE = 0.75   # Minimum masked template score

def run_calibrated_audit():
    # 1. Load All Assets
    bg_templates = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) 
                    for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    
    ore_templates = []
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    print(f"--- Calibrated Audit: D_GATE={D_GATE}, O_GATE={O_GATE} ---")

    for f_num in TARGET_FLOORS:
        files = [f for f in os.listdir(run_path) if f.startswith(f"F{f_num}_")]
        if not files: continue
        
        raw_img = cv2.imread(os.path.join(run_path, files[0]))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi = gray[y1:y2, x1:x2]

            # STAGE 1: Subtraction
            min_diff = min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_templates])
            
            label = f"D:{int(min_diff)}"
            color = (120, 120, 120) # Default Gray (Empty)

            if min_diff > D_GATE:
                # STAGE 2: Masked Search
                best_o = 0
                for t in ore_templates:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if score > best_o: best_o = score
                
                if best_o > O_GATE:
                    color = (0, 255, 0) # Green (Found!)
                    label = f"O:{best_o:.2f}"
                else:
                    color = (0, 0, 255) # Red (Occupied but Unknown)
            
            # High-Contrast Overlay
            cv2.rectangle(raw_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2) # Shadow
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1) # Text

        cv2.imwrite(f"Calibrated_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_calibrated_audit()