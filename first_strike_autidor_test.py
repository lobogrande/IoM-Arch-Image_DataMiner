import cv2
import numpy as np
import os
import json

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261) # (X, Y) of the top-left slot center
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306] # Column X-starts
ROW_0_Y_CENTER = 261 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_grid_dna_v18_2(img_gray):
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X)) + 15
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y)) + 15
        if cy >= img_gray.shape[0] or cx >= img_gray.shape[1]:
            dna += "0"; continue
        roi = img_gray[cy-4:cy+4, cx-4:cx+4]
        dna += "1" if np.mean(roi) > 60 else "0"
    return dna

def get_ore_id_v18_2(img_gray, slot_idx, current_floor, templates):
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    audit_roi = roi[12:, :] if slot_idx in [2, 3] else roi
    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    
    for tier, types in templates['ore'].items():
        if tier not in allowed: continue
        for state in ['act', 'sha']:
            for t_img in types[state]:
                t_audit = t_img[12:, :] if slot_idx in [2, 3] else t_img
                res = cv2.matchTemplate(audit_roi, t_audit, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                if score < 0.80 and state == 'act':
                    for angle in [-5, 5]:
                        M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                        rot_t = cv2.warpAffine(t_audit, M, (t_audit.shape[1], t_audit.shape[0]))
                        score = max(score, cv2.matchTemplate(audit_roi, rot_t, cv2.TM_CCOEFF_NORMED).max())
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v18_2_production_audit():
    buffer_root = "capture_buffer_0"
    out_dir = "production_audit_v18_2"
    for d in ["confirmed", "debug"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
                if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
                if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # ROOT ANCHOR
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    res_init = cv2.matchTemplate(f1_gray[150:480, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
    max_loc_init = cv2.minMaxLoc(res_init)[3]
    init_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc_init[0] - a) <= 12), 0)
    
    anchor = {
        "num": 1, "idx": 0, "pos": max_loc_init, "slot": init_slot,
        "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v18_2(f1_gray, init_slot, 1, ore_tpls),
        "dna": get_grid_dna_v18_2(f1_gray)
    }
    confirmed = []

    print("--- Running v18.2: Strict Row-0 Y-Axis Enforcement ---")

    for i in range(1, len(files)):
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # A. Detect Player (Searching deeper region for row detection)
        res = cv2.matchTemplate(img_gray[150:550, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        # Adjust Y because ROI starts at 150
        actual_y = max_loc[1] + 150 
        
        # B. Analyze State Deltas
        dist = np.sqrt((max_loc[0] - anchor['pos'][0])**2 + (max_loc[1] - anchor['pos'][1])**2)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
        cur_dna = get_grid_dna_v18_2(img_gray)
        dna_diff = sum(1 for a, b in zip(cur_dna, anchor['dna']) if a != b)

        # C. TRIGGER EVALUATION
        commit_reason = None
        if (i - anchor['idx'] > 2):
            if mae > 3.5: commit_reason = "HUD"
            elif dist > 50 and dna_diff >= 4: commit_reason = "STEALTH"

        if commit_reason:
            slot_x = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
            
            # --- THE TRUE ROW-0 FIX ---
            # Player center must be within 25 pixels of the Row 0 Y-center (261)
            is_row_0 = abs(actual_y - ROW_0_Y_CENTER) < 25
            
            if slot_x is not None and is_row_0:
                # COMMIT
                f_num = anchor['num']
                out_img = anchor['img']
                cv2.putText(out_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", out_img)
                confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                
                # NEW ANCHOR
                target_f = f_num + 1
                anchor = {
                    "num": target_f, "idx": i, "pos": max_loc, "slot": slot_x,
                    "img": img_bgr.copy(), "hud": cur_hud.copy(), 
                    "ore": get_ore_id_v18_2(img_gray, slot_x, target_f, ore_tpls),
                    "dna": cur_dna
                }
            else:
                # Debugging Row violations
                debug_label = "not_row_0" if not is_row_0 else "player_lost"
                cv2.imwrite(f"{out_dir}/debug/Idx{i:05}_rej_{debug_label}.jpg", img_bgr)

    # FINAL COMMIT
    if abs(anchor['pos'][1] + 150 - ROW_0_Y_CENTER) < 25:
        cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
        confirmed.append({"floor": anchor['num'], "idx": anchor['idx'], "ore": anchor['ore']})

    with open("Final_FloorMap_v18_2.json", "w") as f:
        json.dump(confirmed, f, indent=4)
    print(f"\n[FINISH] Verified {len(confirmed)} floors. Duplicates rejected by Y-Axis check.")

if __name__ == "__main__":
    run_v18_2_production_audit()