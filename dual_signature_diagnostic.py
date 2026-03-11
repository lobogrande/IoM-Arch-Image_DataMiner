import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 11)
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# THE GATES
D_THRESHOLD = 18    # Minimum difference to be considered 'Occupied'
O_THRESHOLD = 0.80  # Minimum masked template score to be identified

def get_masked_score(roi, t_img, mask):
    """Calculates masked correlation score."""
    res = cv2.matchTemplate(roi, t_img, cv2.TM_CCORR_NORMED, mask=mask)
    _, score, _, _ = cv2.minMaxLoc(res)
    return score

def run_dual_stage_audit():
    # 1. Load Background Templates (for Stage 1)
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    
    # 2. Load Ore Templates (for Stage 2)
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    print(f"--- Dual-Stage Diagnostic: Run {TARGET_RUN} Floors 1-10 ---")

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

            # --- STAGE 1: SUBTRACTION (Occupancy Gate) ---
            min_diff = 999
            for bg in bg_templates:
                diff_score = np.sum(cv2.absdiff(roi, bg)) / (48*48)
                if diff_score < min_diff: min_diff = diff_score
            
            # --- STAGE 2: MASKED IDENTIFICATION ---
            status_color = (150, 150, 150) # Default Gray
            label = f"D:{int(min_diff)}"
            
            if min_diff > D_THRESHOLD:
                best_o = 0
                best_name = ""
                for t in ore_templates:
                    score = get_masked_score(roi, t['img'], mask)
                    if score > best_o:
                        best_o = score
                        best_name = t['name']
                
                if best_o > O_THRESHOLD:
                    status_color = (0, 255, 0) # Green for Ore
                    label = f"O:{best_o:.2f}"
                else:
                    status_color = (0, 0, 255) # Red for 'Occupied but Unidentified'
            
            # Draw Results (White text with black outline for readability)
            cv2.rectangle(raw_img, (x1, y1), (x2, y2), status_color, 1)
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"DualStage_F{f_num}.jpg", raw_img)
        print(f" [+] Processed Floor {f_num}")

if __name__ == "__main__":
    run_dual_stage_audit()