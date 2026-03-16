import cv2
import numpy as np
import os
import json

# --- DEBUG CONTROL ---
TARGET_FLOOR_LIMIT = 25  # Script will stop after this many transitions

# --- THE "STABLE" COORDINATES (v21 REBASE) ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HUD_STAGE_ROI = (54, 74, 103, 138) 
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]

def is_slot_clean_v28(img_gray, slot_idx, row_idx, templates, is_sliver=False, side='left'):
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = int(SLOT1_CENTER[1] + (row_idx * STEP_Y))
    
    if is_sliver:
        # side='left': Right-moving player, check LEFT edge of slot
        # side='right': Left-moving player, check RIGHT edge of slot
        roi = img_gray[cy-24:cy+24, cx-24:cx-12] if side == 'left' else img_gray[cy-24:cy+24, cx+12:cx+24]
    else:
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]

    # Full Tile = 0.70 threshold, Sliver = 0.77
    thresh = 0.77 if is_sliver else 0.70
    for tier in templates['ore']:
        for state in ['act', 'sha']:
            for t_img in templates['ore'][tier][state]:
                # Slice template to match ROI
                t_roi = (t_img[:, :12] if side == 'left' else t_img[:, 36:]) if is_sliver else t_img
                if cv2.matchTemplate(roi, t_roi, cv2.TM_CCOEFF_NORMED).max() > thresh:
                    return False
    return True

def get_ore_id_v28(img_gray, slot_idx, row_idx, current_floor, templates):
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = int(SLOT1_CENTER[1] + (row_idx * STEP_Y))
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    
    # Range limit logic for ores
    best = {'tier': 'empty', 'score': 0.0}
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v28_sanctity_audit():
    buffer_root = "capture_buffer_0"
    out_dir = "production_audit_v28"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    p_right = cv2.imread("templates/player_right.png", 0)
    p_left = cv2.imread("templates/player_left.png", 0)
    
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # --- IMMEDIATE SEED SAVE ---
    f0_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    f0_gray = cv2.cvtColor(f0_bgr, cv2.COLOR_BGR2GRAY)
    f0_hud = f0_gray[HUD_STAGE_ROI[0]:HUD_STAGE_ROI[1], HUD_STAGE_ROI[2]:HUD_STAGE_ROI[3]]
    
    anchor = {
        "num": 1, "idx": 0, "img": f0_bgr.copy(), 
        "hud": f0_hud.copy(), "pos": (0, 0), "ore": "empty"
    }
    # Save F1 immediately so user knows script is working
    cv2.imwrite(f"{out_dir}/confirmed/F001_Idx00000.jpg", anchor['img'])
    confirmed = [{"floor": 1, "idx": 0}]

    print(f"--- Running v28.0: Sanctity Rebase (F1 seeded from {files[0]}) ---")

    for i in range(1, len(files)):
        if len(confirmed) >= TARGET_FLOOR_LIMIT: break

        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. SCAN MODE SELECTION
        # Check if the whole top row is empty
        row1_clean_precheck = all(is_slot_clean_v28(img_gray, s, 0, ore_tpls) for s in range(6))
        
        mode, player_loc, actual_y, tpl = None, None, 0, None
        if not row1_is_empty := row1_clean_precheck:
            # NORMAL MODE: Right-moving
            res = cv2.matchTemplate(img_gray[150:320, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
            v, _, _, loc = cv2.minMaxLoc(res)
            if v > 0.80: mode, player_loc, actual_y, tpl = 'ROW1', loc, loc[1]+150, p_right
        else:
            # FALLBACK MODE: Left-moving (Row 2)
            res = cv2.matchTemplate(img_gray[300:450, 0:480], p_left, cv2.TM_CCOEFF_NORMED)
            v, _, _, loc = cv2.minMaxLoc(res)
            if v > 0.80: mode, player_loc, actual_y, tpl = 'ROW2', loc, loc[1]+300, p_left

        if not mode: continue

        # 2. TRANSITION TRIGGER
        cur_hud = img_gray[HUD_STAGE_ROI[0]:HUD_STAGE_ROI[1], HUD_STAGE_ROI[2]:HUD_STAGE_ROI[3]]
        mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))

        if (i - anchor['idx'] > 5) and (mae > 3.5):
            n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(player_loc[0] - a) <= 15), None)
            if n_slot is None: continue

            # 3. PATH AUDIT (Strict Mirrored Implementation)
            path_is_clean = True
            if mode == 'ROW1':
                # Check Sliver at N-1 (Left), then Full Tiles 0 to N-2
                if n_slot > 0 and not is_slot_clean_v28(img_gray, n_slot-1, 0, ore_tpls, True, 'left'): path_is_clean = False
                if path_is_clean and n_slot >= 2:
                    for s in range(n_slot-1): 
                        if not is_slot_clean_v28(img_gray, s, 0, ore_tpls, False): path_is_clean = False; break
            else: # ROW 2 (Right-to-Left)
                # Check Sliver at N+1 (Right), then Full Tiles N+2 to 5
                if n_slot < 5 and not is_slot_clean_v28(img_gray, n_slot+1, 1, ore_tpls, True, 'right'): path_is_clean = False
                if path_is_clean and n_slot <= 3:
                    for s in range(n_slot+2, 6):
                        if not is_slot_clean_v28(img_gray, s, 1, ore_tpls, False): path_is_clean = False; break

            if path_is_clean:
                f_num = anchor['num']
                target_f = f_num + 1
                r_idx = 0 if mode == 'ROW1' else 1
                
                # COMMIT CURRENT ANCHOR AS PREVIOUS FLOOR
                anchor['ore'] = get_ore_id_v28(img_gray, n_slot, r_idx, target_f, ore_tpls)
                # Save the new anchor as the current floor start
                anchor = {
                    "num": target_f, "idx": i, "pos": (player_loc[0], actual_y),
                    "img": img_bgr.copy(), "hud": cur_hud.copy()
                }
                cv2.imwrite(f"{out_dir}/confirmed/F{target_f:03}_Idx{i:05}.jpg", anchor['img'])
                confirmed.append({"floor": target_f, "idx": i})
                print(f" [OK] F{target_f} ({mode}) locked at Index {i}")

    print(f"\n[FINISH] Verified {len(confirmed)} floors.")

if __name__ == "__main__":
    run_v28_sanctity_audit()