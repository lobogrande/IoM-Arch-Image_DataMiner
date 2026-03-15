import cv2
import numpy as np
import os
import json

# --- PRECISION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31)
}

def get_grid_dna_v17(img_gray):
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

def get_ore_id_v17(img_gray, slot_idx, current_floor, templates):
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
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.76 else "empty"

def run_v17_slot_lock():
    buffer_root = "capture_buffer_0"
    out_dir = "gap_hunter_v17"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None:
                tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
                if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
                if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(cv2.resize(img, (48, 48)))

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    res_init = cv2.matchTemplate(f1_gray[150:420, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
    max_loc_init = cv2.minMaxLoc(res_init)[3]
    init_slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc_init[0] - a) <= 12), 0)
    
    anchor = {
        "num": 1, "idx": 0, "slot": init_slot, "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v17(f1_gray, init_slot, 1, ore_tpls),
        "dna": get_grid_dna_v17(f1_gray)
    }
    confirmed = []

    print("--- Running v17.0 Slot-Lock Precision Auditor ---")

    for i in range(1, 173):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        res = cv2.matchTemplate(img_gray[150:420, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        
        # 1. State Analysis
        cur_slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 15), -1)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
        cur_dna = get_grid_dna_v17(img_gray)
        dna_diff = sum(1 for a, b in zip(cur_dna, anchor['dna']) if a != b)

        # 2. THE PRECISION GATES
        # Gate A: Did the Stage Number change? (Absolute Truth)
        hud_interrupt = (mae > 3.8) 
        
        # Gate B: Did the player move to a NEW slot + World Refresh?
        slot_migration = (cur_slot != anchor['slot'] and cur_slot != -1 and dna_diff >= 3)
        
        # 3. TRIGGER EVALUATION
        if (i - anchor['idx'] > 2) and (hud_interrupt or slot_migration):
            
            # Commit Anchor
            f_num = anchor['num']
            cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
            confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
            print(f" [OK] F{f_num} Locked (Idx {anchor['idx']}, Slot: {anchor['slot']})")
            
            # Update Anchor
            target_f = f_num + 1
            anchor = {
                "num": target_f, "idx": i, "slot": cur_slot, 
                "img": cv2.imread(os.path.join(buffer_root, files[i])), 
                "hud": cur_hud.copy(), 
                "ore": get_ore_id_v17(img_gray, cur_slot, target_f, ore_tpls),
                "dna": cur_dna
            }

    # Final Commit
    cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
    confirmed.append({"floor": anchor['num'], "idx": anchor['idx'], "ore": anchor['ore']})
    print(f"\n[FINISH] Verified {len(confirmed)} floors.")

if __name__ == "__main__":
    run_v17_slot_lock()