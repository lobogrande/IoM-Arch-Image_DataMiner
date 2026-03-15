import cv2
import numpy as np
import os
import json

# --- VERIFIED CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
HEADER_ROI = (54, 74, 103, 138)  # y1, y2, x1, x2
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_MATCH_THRESHOLD = 0.82 
BG_MATCH_THRESHOLD = 0.92
SLIVER_WIDTH = 10 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_best_bg_match(roi, bg_tpls, is_sliver=False):
    """Calibrated sliver audit to detect background behind player's back."""
    best_score = 0
    audit_roi = roi[:, :SLIVER_WIDTH] if is_sliver else roi
    for tpl in bg_tpls:
        audit_tpl = tpl[:, :SLIVER_WIDTH] if is_sliver else tpl
        if audit_roi.shape != audit_tpl.shape:
            audit_tpl = cv2.resize(audit_tpl, (audit_roi.shape[1], audit_roi.shape[0]))
        res = cv2.matchTemplate(audit_roi, audit_tpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        if score > best_score: best_score = score
    return best_score

def get_ore_id_eco(img_gray, slot_idx, current_floor, templates):
    """Template matching filtered by ecological spawn rules."""
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    if roi.shape != (48, 48): roi = cv2.resize(roi, (48, 48))
    
    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    
    for tier, types in templates['ore'].items():
        if tier not in allowed: continue
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.80 else "empty"

def run_v14_unified():
    buffer_root = "capture_buffer_0"
    out_dir = "final_audit_v14"
    for d in ["confirmed", "rejects"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    # 1. ASSET LOADING
    bg_tpls = []
    for f in os.listdir("templates"):
        if "background_plain_" in f:
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None: bg_tpls.append(cv2.resize(img, (48, 48)))

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
    
    # 2. INITIAL ANCHOR SEEDING
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, 
        "img": cv2.imread(os.path.join(buffer_root, files[0])),
        "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "ore": get_ore_id_eco(f1_gray, 0, 1, ore_tpls)
    }
    
    confirmed = []
    current_f = 1

    print("--- Running Unified v14.0 Auditor ---")

    for i in range(1, len(files)):
        if i % 1000 == 0: print(f" Scanning {i} / {len(files)}...", end='\r')
        
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # A. PLAYER DETECTION
        res = cv2.matchTemplate(img_gray[150:400, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)

        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 12), None)
            
            if slot is not None:
                # B. SPATIAL GAP CHECK
                all_clear = True
                # Check distant slots
                for s in range(slot - 1):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    roi = img_gray[261-24:261+24, cx-24:cx+24]
                    if get_best_bg_match(roi, bg_tpls, False) < BG_MATCH_THRESHOLD:
                        all_clear = False; break
                
                # Check overlap sliver (10px behind player)
                if all_clear and slot > 0:
                    cx_overlap = int(SLOT1_CENTER[0] + ((slot-1) * STEP_X))
                    roi_overlap = img_gray[261-24:261+24, cx_overlap-24:cx_overlap+24]
                    if get_best_bg_match(roi_overlap, bg_tpls, True) < BG_MATCH_THRESHOLD:
                        all_clear = False

                # C. VALIDATION & STATE-LOCK
                if all_clear and (i - anchor['idx'] > 2):
                    cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    cur_ore = get_ore_id_eco(img_gray, slot, current_f, ore_tpls)
                    
                    # Fuzzy Identity Check
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
                    if mae < 5.0 and cur_ore == anchor['ore'] and slot == anchor['slot']:
                        continue # REJECT: Pixel-clone of the current anchor

                    # COMMIT TRANSITION
                    f_num = anchor['num']
                    out_img = anchor['img']
                    cx_box = int(SLOT1_CENTER[0] + (anchor['slot'] * STEP_X))
                    cv2.rectangle(out_img, (cx_box-24, 261-24), (cx_box+24, 261+24), (0,255,255), 2)
                    cv2.putText(out_img, f"F{f_num} {anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                    cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{anchor['idx']:05}.jpg", out_img)
                    
                    confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                    print(f"\n [OK] F{f_num} Confirmed at Frame {anchor['idx']} ({anchor['ore']})")

                    # NEW ANCHOR
                    current_f = f_num + 1
                    anchor = {
                        "num": current_f, "idx": i, "slot": slot, 
                        "img": img_bgr.copy(), "hud": cur_hud, "ore": cur_ore
                    }

    # Final cleanup
    with open("Final_FloorMap_v14.json", "w") as f:
        json.dump(confirmed, f, indent=4)
    print(f"\n[FINISH] Auditor complete. {len(confirmed)} floors verified.")

if __name__ == "__main__":
    run_v14_unified()