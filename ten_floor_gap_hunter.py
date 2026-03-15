import cv2
import numpy as np
import os
import json

# --- VERIFIED CALIBRATION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_MATCH_THRESHOLD = 0.80 
BG_MATCH_THRESHOLD = 0.88 

# Ecological restrictions based on stage
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_grid_dna_v16_2(img_gray):
    """Samples the bottom-right corner of each slot to avoid floating damage text."""
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X)) + 15
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y)) + 15
        if cy >= img_gray.shape[0] or cx >= img_gray.shape[1]:
            dna += "0"
            continue
        roi = img_gray[cy-4:cy+4, cx-4:cx+4]
        dna += "1" if np.mean(roi) > 58 else "0"
    return dna

def get_ore_id_v16_2(img_gray, slot_idx, current_floor, templates):
    """Pattern matching for ores with ecological range validation."""
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    
    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    
    for tier, types in templates['ore'].items():
        if tier not in allowed: continue
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                if score > best['score']:
                    best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.78 else "empty"

def run_v16_2_early_hunter():
    buffer_root = "capture_buffer_0"
    out_dir = "gap_hunter_v16_2"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    # 1. ROBUST ASSET LOADING
    bg_tpls = []
    for i in range(8):
        path = f"templates/background_plain_{i}.png"
        img = cv2.imread(path, 0)
        if img is not None:
            bg_tpls.append(cv2.resize(img, (48, 48)))

    player_t = cv2.imread("templates/player_right.png", 0)
    if player_t is None:
        print("ERROR: Could not load player template.")
        return
    
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            raw_img = cv2.imread(os.path.join("templates", f), 0)
            if raw_img is not None:
                tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
                img = cv2.resize(raw_img, (48, 48))
                if tier not in ore_tpls['ore']: 
                    ore_tpls['ore'][tier] = {'act': [], 'sha': []}
                if state in ['act', 'sha']: 
                    ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # 2. STARTING ANCHOR (Floor 1)
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, 
        "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v16_2(f1_gray, 0, 1, ore_tpls),
        "dna": get_grid_dna_v16_2(f1_gray)
    }
    confirmed = []

    print("--- Hunting Floors 1-10 (Indices 0-172) ---")

    # Limit range to solve early-game gaps
    limit = min(173, len(files))
    for i in range(1, limit):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        
        # Player detection logic
        res = cv2.matchTemplate(img_gray[180:360, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)

        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 12), None)
            
            if slot is not None:
                # All-Clear Validation (Spatial Gap)
                all_clear = True
                for s in range(slot):
                    roi = img_gray[261-24:261+24, int(SLOT1_CENTER[0]+s*STEP_X)-24:int(SLOT1_CENTER[0]+s*STEP_X)+24]
                    if cv2.matchTemplate(roi, bg_tpls[0], cv2.TM_CCOEFF_NORMED).max() < BG_MATCH_THRESHOLD:
                        all_clear = False; break

                if all_clear:
                    cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    target_floor = anchor['num'] + 1
                    cur_ore = get_ore_id_v16_2(img_gray, slot, target_floor, ore_tpls)
                    cur_dna = get_grid_dna_v16_2(img_gray)
                    
                    # Persistence & Identity logic
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
                    # We commit if it's a new HUD or new physical Grid state
                    if mae < 4.0 and cur_dna == anchor['dna']:
                        continue 

                    # COMMIT ANCHOR
                    f_num = anchor['num']
                    o_img = anchor['img']
                    cv2.putText(o_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                    cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", o_img)
                    confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                    
                    print(f" [COMMIT] Floor {f_num} found at Index {anchor['idx']}")
                    
                    # Advance to next anchor
                    anchor = {
                        "num": f_num + 1, "idx": i, "slot": slot, 
                        "img": cv2.imread(os.path.join(buffer_root, files[i])), 
                        "hud": cur_hud, "ore": cur_ore, "dna": cur_dna
                    }

    # Save mapping for the sub-range
    with open("EarlyGame_GapMap.json", "w") as f:
        json.dump(confirmed, f, indent=4)
        
    print(f"\n[FINISH] Audit of indices 0-172 complete. {len(confirmed)} floors confirmed.")

if __name__ == "__main__":
    run_v16_2_early_hunter()