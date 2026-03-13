import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 51)
UNIFIED_ROOT = "Unified_Consensus_Inputs"
# Timestamped subfolder for organization
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
OUTPUT_DIR = f"diagnostic_results/Run_{TARGET_RUN}_{TIMESTAMP}"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# THE VALIDATED GATES (v2.7 logic)
D_GATE = 6      
O_GATE = 0.68   
PLAYER_REJECT_GATE = 0.88 
UI_REJECT_GATE = 0.80
DELTA_GATE = 0.05
SUSPICION_THRESHOLD = 0.75 # Ores below this are logged as 'shaky'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_precision_mask(slot_id, mode='ore'):
    mask = np.zeros((48, 48), dtype=np.uint8)
    if mode == 'ore' and slot_id in [1, 2, 3, 4]:
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        cv2.circle(mask, (24, 24), 16, 255, -1)
    return mask

def run_qa_truth_audit():
    # 1. Asset Loading
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    player_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                        for f in os.listdir("templates") if f.startswith("negative_player")]
    ui_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("negative_ui")]
    
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background") or f.startswith("negative"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            # FIX: Extract first prefix {ore type/tier}
            # Handles 'Dirt1_act_none_0.png' -> 'Dirt1'
            clean_name = f.split("_")[0] 
            ore_templates.append({'name': clean_name, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    suspicion_log = []

    print(f"--- Running v2.9.1 QA Audit (Results -> {OUTPUT_DIR}) ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi_gray = gray[y1:y2, x1:x2]

            # GATE 1 & 2: Occupancy/Player
            min_diff = min([np.sum(cv2.absdiff(roi_gray, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue
            
            best_p = max([cv2.matchTemplate(roi_gray, pt, cv2.TM_CCORR_NORMED).max() for pt in player_templates] + [0])
            if best_p > PLAYER_REJECT_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                continue

            # GATE 3: Identification & Comparative Noise Check
            ore_mask = get_precision_mask(slot, mode='ore')
            best_o, best_label = 0, ""
            for t in ore_templates:
                res = cv2.matchTemplate(roi_gray, t['img'], cv2.TM_CCORR_NORMED, mask=ore_mask)
                if res.max() > best_o:
                    best_o, best_label = res.max(), t['name']
            
            best_u = max([cv2.matchTemplate(roi_gray, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_templates] + [0])
            bg_match = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_templates])

            # Double-Verification (v2.7)
            if slot in [1, 2, 3, 4]:
                if (best_u > (best_o + 0.03)) or (best_o < 0.85 and np.max(roi_gray[5:15, :]) > 242):
                    cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    continue

            # SUCCESSFUL IDENTIFICATION
            if best_o > O_GATE and (best_o - bg_match > DELTA_GATE):
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Interior Label Placement
                (w, h), _ = cv2.getTextSize(best_label, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y2-h-4), (x1+w+4, y2-2), (0,0,0), -1) # Contrast backing
                cv2.putText(raw_img, best_label, (x1+3, y2-4), 0, 0.3, (0, 255, 0), 1)

                # Flag Suspicious hits for report
                if best_o < SUSPICION_THRESHOLD or (best_o - bg_match < 0.08):
                    suspicion_log.append({
                        "floor": f_num, "slot": slot, "type": best_label, 
                        "conf": round(float(best_o), 3), "delta": round(float(best_o - bg_match), 3)
                    })

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"QA_F{f_num}.jpg"), raw_img)

    with open(os.path.join(OUTPUT_DIR, "suspicion_report.json"), "w") as f:
        json.dump(suspicion_log, f, indent=4)
    
    print(f" [+] Audit Complete. {len(suspicion_log)} ores flagged in suspicion_report.json.")

if __name__ == "__main__":
    run_qa_truth_audit()