import cv2
import numpy as np
import os
import json

# --- VERIFIED CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_MATCH_THRESHOLD = 0.80 
BG_MATCH_THRESHOLD = 0.88 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31)
}

def get_grid_dna_v16_6(img_gray):
    """Bottom-right sampling to stay clear of damage text."""
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X)) + 15
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y)) + 15
        if cy >= img_gray.shape[0] or cx >= img_gray.shape[1]:
            dna += "0"; continue
        roi = img_gray[cy-4:cy+4, cx-4:cx+4]
        dna += "1" if np.mean(roi) > 58 else "0"
    return dna

def get_ore_id_v16_6(img_gray, slot_idx, current_floor, templates):
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
    return best['tier'] if best['score'] > 0.78 else "empty"

def run_v16_6_player_anchored():
    buffer_root = "capture_buffer_0"
    out_dir = "gap_hunter_v16_6"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    # 1. ASSET LOADING
    bg_tpls = [cv2.resize(cv2.imread(f"templates/background_plain_{i}.png", 0), (48, 48)) for i in range(8)]
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
    
    # SEED ANCHOR
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v16_6(f1_gray, 0, 1, ore_tpls),
        "dna": get_grid_dna_v16_6(f1_gray)
    }
    confirmed = []

    print("--- Running v16.6 Player-Anchored Early Hunter ---")

    for i in range(1, 173):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        
        # TRIGGER 1: PLAYER SPRITE (The Master Anchor)
        res = cv2.matchTemplate(img_gray[150:420, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)

        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 12), None)
            
            if slot is not None:
                # VALIDATION 1: BACKGROUND CHECK (Is this a fresh floor?)
                all_clear = True
                for s in range(slot):
                    roi = img_gray[261-24:261+24, int(SLOT1_CENTER[0]+s*STEP_X)-24:int(SLOT1_CENTER[0]+s*STEP_X)+24]
                    if cv2.matchTemplate(roi, bg_tpls[0], cv2.TM_CCOEFF_NORMED).max() < BG_MATCH_THRESHOLD:
                        all_clear = False; break

                if all_clear:
                    cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    cur_dna = get_grid_dna_v16_6(img_gray)
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))

                    # TRIGGER 2: IDENTITY PERSISTENCE
                    # Commit if the HUD shifted OR the Grid layout changed
                    if (mae > 2.5 or cur_dna != anchor['dna']) and (i - anchor['idx'] > 2):
                        
                        f_num = anchor['num']
                        cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
                        confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                        print(f" [OK] Stage {f_num} Confirmed at Index {anchor['idx']}")
                        
                        target_floor = f_num + 1
                        anchor = {
                            "num": target_floor, "idx": i, "slot": slot, 
                            "img": cv2.imread(os.path.join(buffer_root, files[i])), 
                            "hud": cur_hud, "ore": get_ore_id_v16_6(img_gray, slot, target_floor, ore_tpls),
                            "dna": cur_dna
                        }

    # COMMIT FINAL
    cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
    confirmed.append({"floor": anchor['num'], "idx": anchor['idx'], "ore": anchor['ore']})
    print(f"\n[FINISH] Verified {len(confirmed)} floors.")

if __name__ == "__main__":
    run_v16_6_player_anchored()