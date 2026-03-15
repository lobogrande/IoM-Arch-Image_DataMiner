import cv2
import numpy as np
import os
import json

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]
ROW_0_Y_CENTER = 261 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_grid_dna_fast(img_gray):
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X)) + 15
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y)) + 15
        roi = img_gray[cy-4:cy+4, cx-4:cx+4]
        dna += "1" if np.mean(roi) > 60 else "0"
    return dna

def get_ore_id_fast(img_gray, slot_idx, current_floor, templates):
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
                _, score, _, _ = cv2.minMaxLoc(res)
                
                if score < 0.80 and state == 'act':
                    for angle in [-5, 5]:
                        M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                        rot_t = cv2.warpAffine(t_audit, M, (t_audit.shape[1], t_audit.shape[0]))
                        score = max(score, cv2.matchTemplate(audit_roi, rot_t, cv2.TM_CCOEFF_NORMED).max())
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v18_3_fast_audit():
    buffer_root = "capture_buffer_0"
    out_dir = "production_audit_v18_3"
    for d in ["confirmed", "debug"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # ROOT ANCHOR
    f1_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    res_init = cv2.matchTemplate(f1_gray[150:480, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
    max_loc_init = cv2.minMaxLoc(res_init)[3]
    init_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc_init[0] - a) <= 12), 0)
    
    anchor = {
        "num": 1, "idx": 0, "pos": max_loc_init, "slot": init_slot,
        "img": f1_bgr.copy(),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_fast(f1_gray, init_slot, 1, ore_tpls),
        "dna": get_grid_dna_fast(f1_gray)
    }
    confirmed = []

    print("--- Running v18.3: High-Velocity Production Auditor ---")

    for i in range(1, len(files)):
        if i % 1000 == 0: print(f" Auditing {i}/{len(files)}...", end='\r')
        
        # ONE READ ONLY
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # A. LAZY PLAYER SEEKER (Search Row 0 first for speed)
        # Scan y=200 to y=350 first
        res_fast = cv2.matchTemplate(img_gray[200:350, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        min_v, max_v, min_l, max_loc = cv2.minMaxLoc(res_fast)
        actual_y = max_loc[1] + 200
        
        # If confidence is low, search full grid (The expensive fallback)
        if max_v < 0.70:
            res_full = cv2.matchTemplate(img_gray[150:550, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_loc = cv2.minMaxLoc(res_full)
            actual_y = max_loc[1] + 150

        # B. Deltas
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
        
        # SHORT-CIRCUIT: Skip DNA if HUD hasn't moved at all (Speeds up static frames)
        if mae < 1.0 and abs(max_loc[0] - anchor['pos'][0]) < 5:
            continue

        dist = np.sqrt((max_loc[0] - anchor['pos'][0])**2 + (actual_y - anchor['pos'][1])**2)
        cur_dna = get_grid_dna_fast(img_gray)
        dna_diff = sum(1 for a, b in zip(cur_dna, anchor['dna']) if a != b)

        # C. TRIGGER
        commit_reason = None
        if (i - anchor['idx'] > 2):
            if mae > 3.5: commit_reason = "HUD"
            elif dist > 50 and dna_diff >= 4: commit_reason = "STEALTH"

        if commit_reason:
            slot_x = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
            is_row_0 = abs(actual_y - ROW_0_Y_CENTER) < 25
            
            if slot_x is not None and is_row_0:
                f_num = anchor['num']
                out_img = anchor['img']
                cv2.putText(out_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", out_img)
                confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                
                target_f = f_num + 1
                anchor = {
                    "num": target_f, "idx": i, "pos": (max_loc[0], actual_y), "slot": slot_x,
                    "img": img_bgr, "hud": cur_hud.copy(), 
                    "ore": get_ore_id_fast(img_gray, slot_x, target_f, ore_tpls),
                    "dna": cur_dna
                }
            else:
                if not is_row_0:
                    cv2.imwrite(f"{out_dir}/debug/Idx{i:05}_rej_not_row0.jpg", img_bgr)

    # FINAL COMMIT
    cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
    confirmed.append({"floor": anchor['num'], "idx": anchor['idx'], "ore": anchor['ore']})

    with open("Final_FloorMap_v18_3.json", "w") as f:
        json.dump(confirmed, f, indent=4)
    print(f"\n[FINISH] {len(confirmed)} floors verified. System speed optimized.")

if __name__ == "__main__":
    run_v18_3_fast_audit()