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

# THE VALIDATED GATES
D_GATE = 6      
O_GATE = 0.68   
PLAYER_GATE = 0.88 
DELTA_GATE = 0.05  

def get_precision_mask(slot_id, is_player_check=False):
    mask = np.zeros((48, 48), dtype=np.uint8)
    if not is_player_check and slot_id in [1, 2, 3, 4]:
        # Keep masking the ORE search to ignore UI text
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        cv2.circle(mask, (24, 24), 16, 255, -1)
    return mask

def run_master_precision_audit():
    # 1. Load Assets
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    player_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                        for f in os.listdir("templates") if f.startswith("negative_player")]
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background") or f.startswith("negative"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v1.9.6 Master Precision Suite ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi = gray[y1:y2, x1:x2]
            slot_mask = get_precision_mask(slot)

            # --- GATE 1: OCCUPANCY ---
            min_diff = min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue
            
            # --- GATE 2: PLAYER REJECTION ---
            is_player = False
            p_mask = get_precision_mask(slot, is_player_check=True)
            for pt in player_templates:
                if cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED, mask=p_mask).max() > PLAYER_GATE:
                    is_player = True
                    break
            if is_player:
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                continue

            # --- GATE 3: IDENTIFICATION ---
            best_o = 0
            for t in ore_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=slot_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_o: best_o = score
            
            # Standard background match (No Mask) to maintain the 'Noise Floor'
            bg_match = max([cv2.matchTemplate(roi, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_templates])

            # UI Text Logic for Top Row
            if slot in [1,2,3,4]:
                # If the match is weak AND the top pixels are very bright, it's UI text
                top_zone_brightness = np.mean(roi[5:15, :])
                if best_o < 0.85 and top_zone_brightness > 180:
                    continue # Reject UI text ghost

            if best_o > O_GATE and (best_o - bg_match > DELTA_GATE):
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"O:{best_o:.2f}"
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"Fixed_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_master_precision_audit()