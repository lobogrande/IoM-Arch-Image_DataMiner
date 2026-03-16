import cv2
import numpy as np
import os
import json

# --- DEBUG CONTROL ---
TARGET_FLOOR_LIMIT = 25  # Script stops after this many positive transitions

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]
ROW_0_MAX_Y = 300 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def is_slot_clean_generic(img_gray, slot_idx, templates, is_sliver=False):
    """
    Checks if a slot is empty. 
    If is_sliver is True, it only checks the leftmost 12 pixels (for overlap zone).
    """
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    
    # ROI selection based on sliver requirement
    if is_sliver:
        roi = img_gray[cy-24:cy+24, cx-24:cx-12]
    else:
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]

    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for t_img in types[state]:
                # Dynamic template cropping to match ROI
                t_roi = t_img[:, :12] if is_sliver else t_img
                res = cv2.matchTemplate(roi, t_roi, cv2.TM_CCOEFF_NORMED)
                if cv2.minMaxLoc(res)[1] > 0.77:
                    return False # Found an ore/shadow
    return True

def get_ore_id_v21(img_gray, slot_idx, current_floor, templates):
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
                # Dynamic template cropping for slot 2/3 comparison
                t_audit = t_img[12:, :] if slot_idx in [2, 3] else t_img
                res = cv2.matchTemplate(audit_roi, t_audit, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v35_authentic_rebase():
    buffer_root = "capture_buffer_0"
    out_dir = "production_audit_v35"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)

    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {'ore': {}}
    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # ROOT ANCHOR
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    res_init = cv2.matchTemplate(f1_gray[150:480, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
    max_loc_init = cv2.minMaxLoc(res_init)[3]
    
    anchor = {
        "num": 1, "idx": 0, "pos": (max_loc_init[0], max_loc_init[1] + 150),
        "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v21(f1_gray, 0, 1, ore_tpls)
    }
    confirmed = []

    print(f"--- Running v35.0: Authentic v21 Rebase with Floor Cap: {TARGET_FLOOR_LIMIT} ---")

    for i in range(1, len(files)):
        # Exit condition for targeted debugging
        if len(confirmed) >= TARGET_FLOOR_LIMIT:
            break

        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Primary Gate (Row 0 Only)
        res = cv2.matchTemplate(img_gray[150:500, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        actual_y = max_loc[1] + 150
        if actual_y > ROW_0_MAX_Y: continue

        # 2. Secondary Gate (HUD Shift)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        if np.mean(cv2.absdiff(cur_hud, anchor['hud'])) < 3.5: continue

        # 3. SYSTEMATIC N-RELATIVE PATH AUDIT
        n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
        if n_slot is None: continue

        path_is_clean = True
        
        # A. AUDIT N-1 (The Sliver Zone)
        if n_slot > 0:
            if not is_slot_clean_generic(img_gray, n_slot - 1, ore_tpls, is_sliver=True):
                path_is_clean = False

        # B. AUDIT 0 to N-2 (The Empty Zone)
        if path_is_clean and n_slot >= 2:
            for s_idx in range(n_slot - 1): # Range stops at N-2
                if not is_slot_clean_generic(img_gray, s_idx, ore_tpls, is_sliver=False):
                    path_is_clean = False; break
        
        if path_is_clean:
            # COMMIT PREVIOUS
            f_num = anchor['num']
            out_img = anchor['img']
            cv2.putText(out_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
            cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", out_img)
            confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
            
            # UPDATE ANCHOR
            target_f = f_num + 1
            anchor = {
                "num": target_f, "idx": i, "pos": (max_loc[0], actual_y),
                "img": img_bgr.copy(), "hud": cur_hud.copy(), 
                "ore": get_ore_id_v21(img_gray, n_slot, target_f, ore_tpls)
            }
            print(f" [OK] Stage {target_f} locked at Index {i} (Player Slot {n_slot})")

    # FINAL COMMIT
    cv2.imwrite(f"{out_dir}/confirmed/F{anchor['num']:03}_Idx{anchor['idx']:05}.jpg", anchor['img'])
    print(f"\n[FINISH] Verified {len(confirmed)+1} floors.")

if __name__ == "__main__":
    run_v35_authentic_rebase()