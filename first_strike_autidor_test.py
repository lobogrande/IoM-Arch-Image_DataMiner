import cv2
import numpy as np
import os
import json

# --- CALIBRATED PROJECT CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_MATCH_THRESHOLD = 0.82 
BG_MATCH_THRESHOLD = 0.92
SLIVER_WIDTH = 10 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_grid_dna(img_gray):
    """Generates a 24-bit pattern of filled/empty states."""
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X))
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        dna += "1" if np.mean(roi) > 55 else "0"
    return dna

def get_best_bg_match(roi, bg_tpls, is_sliver=False):
    best_score = 0
    audit_roi = roi[:, :SLIVER_WIDTH] if is_sliver else roi
    for tpl in bg_tpls:
        audit_tpl = tpl[:, :SLIVER_WIDTH] if is_sliver else tpl
        res = cv2.matchTemplate(audit_roi, audit_tpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        if score > best_score: best_score = score
    return best_score

def get_ore_id_v15(img_gray, slot_idx, current_floor, templates):
    """
    Ore ID with Hit-Rotation Compensation and Slot 2/3 text masking.
    """
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    
    # ROI: 48x48 centered on slot
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    
    # MASKING: If Slot 2 or 3, we ignore the top 12 pixels where 'Dig Stage' text sits
    if slot_idx in [2, 3]:
        audit_roi = roi[12:, :]
    else:
        audit_roi = roi

    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    
    for tier, types in templates['ore'].items():
        if tier not in allowed: continue
        for state in ['act', 'sha']:
            for t_img in types[state]:
                t_audit = t_img[12:, :] if slot_idx in [2, 3] else t_img
                
                # BASE PASS
                res = cv2.matchTemplate(audit_roi, t_audit, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(res)[1]
                
                # ROTATION PASS (Only for active ores during hit animation)
                if score < 0.80 and state == 'act':
                    for angle in [-5, 5]:
                        M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                        rot_t = cv2.warpAffine(t_audit, M, (t_audit.shape[1], t_audit.shape[0]))
                        res_rot = cv2.matchTemplate(audit_roi, rot_t, cv2.TM_CCOEFF_NORMED)
                        score = max(score, cv2.minMaxLoc(res_rot)[1])

                if score > best['score']:
                    best = {'tier': tier, 'score': score}
                    
    return best['tier'] if best['score'] > 0.78 else "empty"

def run_v15_unified():
    buffer_root = "capture_buffer_0"
    out_dir = "final_audit_v15"
    for d in ["confirmed", "rejects"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    # Asset Loading
    bg_tpls = []
    for f in os.listdir("templates"):
        if "background_plain_" in f: bg_tpls.append(cv2.resize(cv2.imread(f"templates/{f}", 0), (48, 48)))
    player_t = cv2.imread("templates/player_right.png", 0)
    
    ore_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): ore_tpls['bg'].append(img)
        elif "_" in f:
            parts = f.split("_")
            tier, state = parts[0], parts[1]
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # Initial Anchor
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_v15(f1_gray, 0, 1, ore_tpls),
        "dna": get_grid_dna(f1_gray)
    }
    confirmed = []
    current_f = 1

    print("--- Running v15.0 Resilient Auditor ---")
    for i in range(1, len(files)):
        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        res = cv2.matchTemplate(img_gray[150:400, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)

        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 12), None)
            if slot is not None:
                all_clear = True
                for s in range(slot - 1):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    if get_best_bg_match(img_gray[261-24:261+24, cx-24:cx+24], bg_tpls) < BG_MATCH_THRESHOLD:
                        all_clear = False; break
                
                if all_clear and slot > 0:
                    cx_overlap = int(SLOT1_CENTER[0] + ((slot-1) * STEP_X))
                    if get_best_bg_match(img_gray[261-24:261+24, cx_overlap-24:cx_overlap+24], bg_tpls, True) < BG_MATCH_THRESHOLD:
                        all_clear = False

                if all_clear and (i - anchor['idx'] > 2):
                    cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    cur_ore = get_ore_id_v15(img_gray, slot, current_f, ore_tpls)
                    cur_dna = get_grid_dna(img_gray)
                    
                    # PERSISTENCE CHECK (HUD OR DNA)
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
                    is_clone = (mae < 5.0 or cur_dna == anchor['dna']) and (cur_ore == anchor['ore'] or cur_ore == "empty")

                    if is_clone:
                        continue 

                    # COMMIT
                    f_num, o_img = anchor['num'], anchor['img']
                    cx_box = int(SLOT1_CENTER[0] + (anchor['slot'] * STEP_X))
                    cv2.rectangle(o_img, (cx_box-24, 261-24), (cx_box+24, 261+24), (0,255,255), 2)
                    cv2.putText(o_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                    cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", o_img)
                    confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                    
                    print(f" [OK] F{f_num} Locked ({anchor['ore']})")
                    current_f = f_num + 1
                    anchor = {"num": current_f, "idx": i, "slot": slot, "img": cv2.imread(os.path.join(buffer_root, files[i])), "hud": cur_hud, "ore": cur_ore, "dna": cur_dna}

    with open("Final_FloorMap_v15.json", "w") as f: json.dump(confirmed, f, indent=4)

if __name__ == "__main__":
    run_v14_unified = run_v15_unified()