import cv2
import numpy as np
import os
import json

# --- WIDE-OPEN CALIBRATION ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]

# Lowered thresholds for maximum sensitivity
PLAYER_MATCH_THRESHOLD = 0.75 
BG_MATCH_THRESHOLD = 0.70 
SNAP_THRESHOLD = 6.0 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74)
}

def get_ore_id_v16_4(img_gray, slot_idx, current_floor, templates):
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
    return best['tier'] if best['score'] > 0.75 else "empty"

def run_v16_4_wide_open():
    buffer_root = "capture_buffer_0"
    out_dir = "gap_hunter_v16_4"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    # 1. ASSET LOADING
    bg_tpls = []
    for i in range(8):
        path = f"templates/background_plain_{i}.png"
        img = cv2.imread(path, 0)
        if img is not None: bg_tpls.append(cv2.resize(img, (48, 48)))

    player_t = cv2.imread("templates/player_right.png", 0)
    
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            raw_img = cv2.imread(os.path.join("templates", f), 0)
            if raw_img is not None:
                tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
                img = cv2.resize(raw_img, (48, 48))
                if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
                if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # SEED FIRST ANCHOR
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, 
        "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v16_4(f1_gray, 0, 1, ore_tpls)
    }
    
    confirmed = []
    prev_gray = f1_gray.copy()

    print("--- Running v16.4 Wide-Open Early Hunter ---")

    for i in range(1, 173):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        frame_diff = np.mean(cv2.absdiff(img_gray, prev_gray))
        
        res = cv2.matchTemplate(img_gray[150:420, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)

        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
            
            if slot is not None:
                # RELAXED SPATIAL CHECK
                all_clear = True
                for s in range(slot):
                    roi = img_gray[261-24:261+24, int(SLOT1_CENTER[0]+s*STEP_X)-24:int(SLOT1_CENTER[0]+s*STEP_X)+24]
                    if cv2.matchTemplate(roi, bg_tpls[0], cv2.TM_CCOEFF_NORMED).max() < BG_MATCH_THRESHOLD:
                        all_clear = False; break

                if all_clear:
                    cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
                    
                    # WIDE-OPEN TRIGGER: Just a screen pulse or a HUD change
                    if (frame_diff > SNAP_THRESHOLD or mae > 2.0) and (i - anchor['idx'] > 2):
                        
                        # COMMIT COMPLETED FLOOR
                        f_num = anchor['num']
                        cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
                        confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                        print(f" [OK] F{f_num} at Index {anchor['idx']} ({anchor['ore']})")
                        
                        # SET NEW ANCHOR
                        target_floor = f_num + 1
                        anchor = {
                            "num": target_floor, "idx": i, "slot": slot, 
                            "img": cv2.imread(os.path.join(buffer_root, files[i])), 
                            "hud": cur_hud, "ore": get_ore_id_v16_4(img_gray, slot, target_floor, ore_tpls)
                        }

        prev_gray = img_gray.copy()

    # COMMIT FINAL
    cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
    confirmed.append({"floor": anchor['num'], "idx": anchor['idx'], "ore": anchor['ore']})

    print(f"\n[FINISH] Found {len(confirmed)} floors.")

if __name__ == "__main__":
    run_v16_4_wide_open()