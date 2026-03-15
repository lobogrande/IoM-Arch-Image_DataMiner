import cv2
import numpy as np
import os
import json

# --- VERIFIED CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def get_grid_dna(img_gray):
    """Generates a 24-bit pattern representing the filled/empty state of the grid."""
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X))
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
        # Sample center of slot
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        # 1 if occupied (ore present), 0 if empty
        dna += "1" if np.mean(roi) > 55 else "0"
    return dna

def get_hud_pixels(img_gray):
    return img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]

def get_ore_id(img_gray, slot_idx, templates, mask):
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    if roi.shape != (48, 48): roi = cv2.resize(roi, (48, 48))
    best_ore = {'tier': 'empty', 'score': 0.0}
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCORR_NORMED, mask=mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_ore['score']:
                    best_ore = {'tier': tier, 'score': score}
    return best_ore['tier'] if best_ore['score'] > 0.80 else "empty"

def run_v11_9_dna_audit():
    mask = get_spatial_mask()
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(f"templates/{f}", 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): ore_tpls['bg'].append(img)
        elif "_" in f:
            parts = f.split("_")
            tier, state = parts[0], parts[1]
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    BUFFER_ROOT = "capture_buffer_0"
    OUT = "diagnostic_results/DNA_v11_9"
    for d in ["confirmed", "rejects"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    # 1. INITIAL ANCHOR
    f1_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, "img": cv2.imread(os.path.join(BUFFER_ROOT, files[0])),
        "hud": get_hud_pixels(f1_gray),
        "ore": get_ore_id(f1_gray, 0, ore_tpls, mask),
        "dna": get_grid_dna(f1_gray)
    }

    confirmed = []
    print("--- Running v11.9 DNA-Enhanced Auditor ---")

    for i in range(1, len(files)):
        if i % 1000 == 0: print(f" [Scan] {i:05}...", end='\r')
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img_gray[180:360, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 4), None)
            if slot is not None:
                # Shadow Check
                is_spawn = True
                for s in range(slot):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    if np.mean(img_gray[261-7:261+7, cx-7:cx+7]) > 55: is_spawn = False; break
                
                if is_spawn and (i - anchor['idx'] > 2):
                    cur_hud = get_hud_pixels(img_gray)
                    cur_ore = get_ore_id(img_gray, slot, ore_tpls, mask)
                    cur_dna = get_grid_dna(img_gray)
                    
                    mae = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
                    
                    # REJECTION VOTE
                    # Vote 1: HUD is identical (Fuzzy)
                    hud_match = (mae < 5.0)
                    # Vote 2: Grid DNA is identical
                    dna_match = (cur_dna == anchor['dna'])
                    # Vote 3: Ore ID is same
                    ore_match = (cur_ore == anchor['ore'])

                    # REJECT if Slot is same AND (HUD matches OR Grid DNA matches)
                    if slot == anchor['slot'] and ore_match and (hud_match or dna_match):
                        reason = "HUD" if hud_match else "DNA"
                        canvas = np.hstack((cv2.resize(anchor['hud'], (160, 100)), cv2.resize(cur_hud, (160, 100))))
                        cv2.putText(canvas, f"REJECT: {reason}", (10, 20), 0, 0.4, (0,0,255), 1)
                        cv2.imwrite(f"{OUT}/rejects/Reject_Idx_{i:05}.jpg", canvas)
                        continue 

                    else:
                        # STATE CHANGE CONFIRMED
                        f_num = anchor['num']
                        out_img = anchor['img']
                        cx_box = int(SLOT1_CENTER[0] + (anchor['slot'] * STEP_X))
                        cv2.rectangle(out_img, (cx_box-24, 261-24), (cx_box+24, 261+24), (0,255,255), 2)
                        cv2.putText(out_img, f"Ore:{anchor['ore']}", (20, 50), 0, 0.7, (255,255,255), 2)
                        cv2.imwrite(f"{OUT}/confirmed/Floor_{f_num:03}_Frame_{anchor['idx']:05}.jpg", out_img)
                        
                        confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                        print(f" [COMMIT] F{f_num} DNA: {anchor['dna'][:8]}...")

                        anchor = {
                            "num": f_num + 1, "idx": i, "slot": slot, "img": img_bgr.copy(),
                            "hud": cur_hud, "ore": cur_ore, "dna": cur_dna
                        }

    with open("Run_0_FloorMap_v11_9.json", 'w') as f:
        json.dump(confirmed, f, indent=4)

if __name__ == "__main__":
    run_v11_9_dna_audit()