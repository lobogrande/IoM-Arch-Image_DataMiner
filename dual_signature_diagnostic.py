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

# THE PRECISION GATES
D_GATE = 8      # Raised from 6 to kill the last background 'ghosts'
O_GATE = 0.72   
PLAYER_REJECT_GATE = 0.88 # Raised significantly to stop ores being called 'player'

def get_precision_mask(slot_id, is_player_check=False):
    """Uses a tight circular mask to ignore background gravel noise."""
    mask = np.zeros((48, 48), dtype=np.uint8)
    
    # Static UI text masking for top row
    if not is_player_check and slot_id in [1, 2, 3, 4]:
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        # Standard central circle (Tightened to 16px radius)
        # This isolates the 'Heart' of the ore/player and ignores the floor corners
        cv2.circle(mask, (24, 24), 16, 255, -1)
    return mask

def run_silhouette_audit():
    # 1. Load Backgrounds
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    
    # 2. Load Player Templates
    player_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                        for f in os.listdir("templates") if f.startswith("negative_player")]

    # 3. Load Ore Templates
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background") or f.startswith("negative"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v1.9.3 Silhouette Rejection Suite ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi = gray[y1:y2, x1:x2]

            # --- GATE 1: OCCUPANCY (Strict) ---
            min_diff = min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue
            
            # --- GATE 2: PLAYER REJECTION (Tight Silhouette) ---
            is_player = False
            p_mask = get_precision_mask(slot, is_player_check=True)
            for pt in player_templates:
                p_res = cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED, mask=p_mask)
                if p_res.max() > PLAYER_REJECT_GATE:
                    is_player = True
                    break
            
            if is_player:
                # Magenta only if it's DEFINITELY the player icon
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                continue

            # --- GATE 3: ORE IDENTIFICATION ---
            best_o = 0
            slot_mask = get_precision_mask(slot)
            for t in ore_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=slot_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_o: best_o = score
            
            if best_o > O_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(raw_img, f"O:{best_o:.2f}", (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
                cv2.putText(raw_img, f"O:{best_o:.2f}", (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"Silhouette_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_silhouette_audit()