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
DELTA_GATE = 0.05  

def get_precision_mask(slot_id, is_player_check=False):
    """Provides targeted spatial masking to ignore UI overlays."""
    mask = np.zeros((48, 48), dtype=np.uint8)
    if not is_player_check and slot_id in [1, 2, 3, 4]:
        # Ignore top 18 pixels specifically for UI text (Slots 1-4)
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        # Standard central circle for all other identifications
        cv2.circle(mask, (24, 24), 16, 255, -1)
    return mask

def is_white_ui_text(roi_color):
    """
    Surgically identifies white UI text while ignoring yellow XP modifiers.
    Uses height-profile analysis and peak intensity.
    """
    # Focus on the 'Dig Stage' overlay zone
    ui_zone_color = roi_color[5:18, 5:43]
    ui_zone_gray = cv2.cvtColor(ui_zone_color, cv2.COLOR_BGR2GRAY)
    
    # 1. Peak Intensity Check (Ores/XP are rarely pure white > 235)
    max_val = np.max(ui_zone_gray)
    if max_val < 230: 
        return False
    
    # 2. Stroke Height Analysis
    _, thresh = cv2.threshold(ui_zone_gray, 220, 255, cv2.THRESH_BINARY)
    column_heights = np.sum(thresh > 0, axis=0)
    active_columns = column_heights[column_heights > 0]
    
    if len(active_columns) == 0: 
        return False
        
    avg_height = np.mean(active_columns)
    
    # UI text letters (D, i, g, S, t, a, g, e) are consistently 7-11px tall.
    # XP icons and Ore glints are typically shorter (<5px) or irregular.
    return 7 <= avg_height <= 11

def run_full_surgical_audit():
    # 1. Load Templates
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

    print(f"--- Running v1.9.9 Master Surgical Auditor ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi_color = raw_img[y1:y2, x1:x2]
            roi_gray = gray[y1:y2, x1:x2]

            # --- GATE 1: OCCUPANCY ---
            min_diff = min([np.sum(cv2.absdiff(roi_gray, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue
            
            # --- GATE 2: PLAYER REJECTION ---
            is_player = False
            p_mask = get_precision_mask(slot, is_player_check=True)
            for pt in player_templates:
                p_res = cv2.matchTemplate(roi_gray, pt, cv2.TM_CCORR_NORMED, mask=p_mask)
                if p_res.max() > PLAYER_GATE:
                    is_player = True
                    break
            
            if is_player:
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                continue

            # --- GATE 3: WHITE UI TEXT SUPPRESSION ---
            # Specifically targets 'Dig Stage' text in Slots 1-4 while protecting yellow XP
            if slot in [1, 2, 3, 4] and is_white_ui_text(roi_color):
                continue

            # --- GATE 4: ORE IDENTIFICATION ---
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

        cv2.imwrite(f"Surgical_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_full_surgical_audit()