import cv2
import numpy as np
import os
import json

# --- DEBUG CONTROL ---
MAX_FLOORS_TO_AUDIT = 15  # Script will stop after identifying this many floors

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]

# Y-centers for spawn rows
ROW_1_Y = 261
ROW_2_Y = 320 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def is_slot_clean_bidirectional(img_gray, slot_idx, row_idx, templates, is_sliver=False, side='left'):
    """
    row_idx: 0 for top row, 1 for second row.
    side: 'left' for right-moving player, 'right' for left-moving player.
    """
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = int(SLOT1_CENTER[1] + (row_idx * STEP_Y))
    
    if is_sliver:
        # Check either the far left or far right 12 pixels to avoid sprite overlap
        roi = img_gray[cy-24:cy+24, cx-24:cx-12] if side == 'left' else img_gray[cy-24:cy+24, cx+12:cx+24]
    else:
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]

    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for t_img in types[state]:
                t_roi = (t_img[:, :12] if side == 'left' else t_img[:, 36:]) if is_sliver else t_img
                res = cv2.matchTemplate(roi, t_roi, cv2.TM_CCOEFF_NORMED)
                if cv2.minMaxLoc(res)[1] > 0.77:
                    return False
    return True

def get_ore_id_v22(img_gray, slot_idx, row_idx, current_floor, templates):
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = int(SLOT1_CENTER[1] + (row_idx * STEP_Y))
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    for tier, types in templates['ore'].items():
        if tier not in allowed: continue
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v22_deep_fallback():
    buffer_root = "capture_buffer_0"
    out_dir = "production_audit_v22"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    p_right = cv2.imread("templates/player_right.png", 0)
    p_left = cv2.imread("templates/player_left.png", 0) # Assumes you have this
    
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # SEED ANCHOR
    anchor = {"num": 1, "idx": 0, "img": cv2.imread(os.path.join(buffer_root, files[0])), "hud": None, "ore": "empty"}
    confirmed = []

    print(f"--- Running v22.0: Capping at Floor {MAX_FLOORS_TO_AUDIT} ---")

    for i in range(1, len(files)):
        if len(confirmed) >= MAX_FLOORS_TO_AUDIT: break # THE CAP

        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. SCAN ROW 1 (Standard)
        res1 = cv2.matchTemplate(img_gray[200:320, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
        v1, _, _, loc1 = cv2.minMaxLoc(res1)
        
        # 2. SCAN ROW 2 (Fallback)
        res2 = cv2.matchTemplate(img_gray[300:400, 0:480], p_left, cv2.TM_CCOEFF_NORMED)
        v2, _, _, loc2 = cv2.minMaxLoc(res2)

        # Selection Logic
        mode, player_loc, actual_y = None, None, 0
        if v1 > 0.80 and (loc1[1]+200) < 300:
            mode, player_loc, actual_y = 'ROW1', loc1, loc1[1]+200
        elif v2 > 0.80 and (loc2[1]+300) < 360:
            # Special case: check if Row 1 is truly empty before allowing Row 2 fallback
            row1_clean = all(is_slot_clean_bidirectional(img_gray, s, 0, ore_tpls) for s in range(6))
            if row1_clean:
                mode, player_loc, actual_y = 'ROW2', loc2, loc2[1]+300
        
        if not mode: continue

        # HUD & Path Logic
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        if anchor['hud'] is not None and np.mean(cv2.absdiff(cur_hud, anchor['hud'])) < 3.5: continue

        n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(player_loc[0] - a) <= 15), None)
        if n_slot is None: continue

        path_is_clean = True
        if mode == 'ROW1': # Right moving
            if n_slot > 0 and not is_slot_clean_bidirectional(img_gray, n_slot-1, 0, ore_tpls, True, 'left'): path_is_clean = False
            if path_is_clean and n_slot >= 2:
                for s in range(n_slot-1): 
                    if not is_slot_clean_bidirectional(img_gray, s, 0, ore_tpls): path_is_clean = False; break
        else: # ROW2: Left moving
            if n_slot < 5 and not is_slot_clean_bidirectional(img_gray, n_slot+1, 1, ore_tpls, True, 'right'): path_is_clean = False
            if path_is_clean and n_slot <= 3:
                for s in range(n_slot+2, 6):
                    if not is_slot_clean_bidirectional(img_gray, s, 1, ore_tpls): path_is_clean = False; break

        if path_is_clean:
            f_num = anchor['num']
            out_img = anchor['img']
            cv2.putText(out_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
            cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", out_img)
            confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
            
            target_f = f_num + 1
            row_idx = 0 if mode == 'ROW1' else 1
            anchor = {
                "num": target_f, "idx": i, "pos": (player_loc[0], actual_y),
                "img": img_bgr.copy(), "hud": cur_hud.copy(), 
                "ore": get_ore_id_v22(img_gray, n_slot, row_idx, target_f, ore_tpls)
            }
            print(f" [OK] F{target_f} ({mode}) found at Index {i}")

    print(f"\n[STOP] Reached floor limit or end of data. {len(confirmed)} floors verified.")

if __name__ == "__main__":
    run_v22_deep_fallback()