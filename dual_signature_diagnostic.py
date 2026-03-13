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

# THE FINAL TUNING
D_GATE = 6      
O_GATE = 0.68   
PLAYER_REJECT_GATE = 0.85 
UI_REJECT_GATE = 0.82
DELTA_GATE = 0.05  

def get_precision_mask(slot_id, mode='ore'):
    mask = np.zeros((48, 48), dtype=np.uint8)
    if mode == 'ore' and slot_id in [1, 2, 3, 4]:
        # Ignore top 18 pixels specifically for UI text (Slots 1-4)
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        # Wider circle for negative checks to ensure we see the whole character/text
        cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def run_mutual_exclusion_audit():
    # 1. Load Assets
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
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v2.1 Mutual Exclusion Auditor ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi_gray = gray[y1:y2, x1:x2]

            # --- GATE 1: OCCUPANCY ---
            min_diff = min([np.sum(cv2.absdiff(roi_gray, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue
            
            # --- GATE 2: POSITIVE ORE SEARCH ---
            best_o = 0
            ore_mask = get_precision_mask(slot, mode='ore')
            for t in ore_templates:
                res = cv2.matchTemplate(roi_gray, t['img'], cv2.TM_CCORR_NORMED, mask=ore_mask)
                score = res.max()
                if score > best_o: best_o = score
            
            # --- GATE 3: NEGATIVE NOISE SEARCH ---
            best_p = max([cv2.matchTemplate(roi_gray, pt, cv2.TM_CCORR_NORMED).max() for pt in player_templates] + [0])
            best_u = max([cv2.matchTemplate(roi_gray, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_templates] + [0])

            # --- FINAL DECISION TREE ---
            # 1. Player Check (Highest Priority)
            if best_p > PLAYER_REJECT_GATE and best_p > best_o:
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1) # Magenta
                continue

            # 2. UI Text Check (Slot 1-4 specific)
            if slot in [1, 2, 3, 4]:
                # If it looks like UI text and contains pure white, reject it
                if best_u > UI_REJECT_GATE or np.max(roi_gray[5:15, :]) > 240:
                    cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 255, 0), 1) # Cyan
                    continue

            # 3. Valid Ore Check
            bg_match = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_templates])
            if best_o > O_GATE and (best_o - bg_match > DELTA_GATE):
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green
                cv2.putText(raw_img, f"O:{best_o:.2f}", (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"Exclusion_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_mutual_exclusion_audit()