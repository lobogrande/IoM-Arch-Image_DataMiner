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

# CORE GATES
D_GATE = 6      
O_GATE = 0.68   
PLAYER_GATE = 0.88 
UI_TEXT_GATE = 0.82  # Sensitivity for the 'Dig Stage' text templates
DELTA_GATE = 0.05  

def get_precision_mask(slot_id, is_negative_check=False):
    mask = np.zeros((48, 48), dtype=np.uint8)
    if not is_negative_check and slot_id in [1, 2, 3, 4]:
        # Ignore top 18 pixels specifically for UI text (Slots 1-4) during ORE search
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        # Full circle for Player/UI rejection to ensure we see the 'noise'
        cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def run_dual_negative_audit():
    # 1. Load Backgrounds
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    
    # 2. Load Negative Templates (Player AND UI Text)
    neg_templates = []
    for f in os.listdir("templates"):
        if f.startswith("negative_player") or f.startswith("negative_ui"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None:
                is_player = f.startswith("negative_player")
                neg_templates.append({
                    'img': cv2.resize(img, (48, 48)), 
                    'threshold': PLAYER_GATE if is_player else UI_TEXT_GATE,
                    'type': 'player' if is_player else 'ui'
                })

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

    print(f"--- Running v2.0 Dual-Negative Auditor ---")

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
            
            # --- GATE 2: DUAL-NEGATIVE REJECTION (Player & UI Text) ---
            is_rejected = False
            neg_mask = get_precision_mask(slot, is_negative_check=True)
            for nt in neg_templates:
                res = cv2.matchTemplate(roi_gray, nt['img'], cv2.TM_CCORR_NORMED, mask=neg_mask)
                if res.max() > nt['threshold']:
                    is_rejected = True
                    # Draw rejection indicator
                    color = (255, 0, 255) if nt['type'] == 'player' else (255, 255, 0) # Cyan for UI Text
                    cv2.rectangle(raw_img, (x1, y1), (x2, y2), color, 1)
                    break
            
            if is_rejected: continue

            # --- GATE 3: ORE IDENTIFICATION ---
            best_o = 0
            slot_mask = get_precision_mask(slot)
            for t in ore_templates:
                res = cv2.matchTemplate(roi_gray, t['img'], cv2.TM_CCORR_NORMED, mask=slot_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_o: best_o = score
            
            bg_match = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_templates])

            if best_o > O_GATE and (best_o - bg_match > DELTA_GATE):
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"O:{best_o:.2f}"
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"DualNeg_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_dual_negative_audit()