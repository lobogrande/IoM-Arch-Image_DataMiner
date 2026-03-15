import cv2
import numpy as np
import os
import json

# --- HARD CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X = 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def get_hud_state(img_gray):
    roi = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    return roi.tobytes(), roi

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

def run_v11_4_anchor_lock():
    # Assets
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
    OUT = "diagnostic_results/AnchorLock_v11_4"
    for d in ["confirmed", "rejects"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
    
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    # 1. INITIAL ANCHOR (Floor 1)
    f1_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)
    f1_bytes, f1_roi = get_hud_state(f1_gray)
    anchor = {
        "num": 1, "idx": 0, "slot": 0, "img": cv2.imread(os.path.join(BUFFER_ROOT, files[0])),
        "hud_bytes": f1_bytes, "hud_visual": f1_roi,
        "ore": get_ore_id(f1_gray, 0, ore_tpls, mask)
    }

    confirmed = []
    diff_frames_count = 0  # Counter for persistence
    print("--- Running v11.4 Immutable Anchor Auditor ---")

    for i in range(1, len(files)):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # STEP A: CANDIDATE CHECK
        res = cv2.matchTemplate(img_gray[200:350, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 3), None)
            if slot is not None:
                # Shadow Check
                is_spawn = True
                for s in range(slot):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    if np.mean(img_gray[261-7:261+7, cx-7:cx+7]) > 55: is_spawn = False; break
                
                if is_spawn and (i - anchor['idx'] > 4):
                    cur_bytes, cur_roi = get_hud_state(img_gray)
                    cur_ore = get_ore_id(img_gray, slot, ore_tpls, mask)
                    
                    # STEP B: THE ANCHOR COMPARISON
                    # We compare EVERYTHING to the 'Floor Anchor', not the most recent find.
                    is_identical = (cur_bytes == anchor['hud_bytes'] and cur_ore == anchor['ore'] and slot == anchor['slot'])

                    if is_identical:
                        # DISCARD: It matches the master floor anchor
                        diff_frames_count = 0 
                        canvas = np.hstack((cv2.resize(anchor['hud_visual'], (160, 100)), cv2.resize(cur_roi, (160, 100))))
                        cv2.putText(canvas, "ANCHOR CLONE", (10, 20), 0, 0.5, (0,0,255), 1)
                        cv2.imwrite(f"{OUT}/rejects/Reject_Idx_{i:05}.jpg", canvas)
                        continue

                    else:
                        # STEP C: PERSISTENCE FILTER
                        # A floor must be "Different" for 3 frames before we trust it.
                        diff_frames_count += 1
                        if diff_frames_count >= 3:
                            # COMMIT PREVIOUS ANCHOR
                            f_num = anchor['num']
                            out_img = anchor['img']
                            cx_box = int(SLOT1_CENTER[0] + (anchor['slot'] * STEP_X))
                            cv2.rectangle(out_img, (cx_box-24, 261-24), (cx_box+24, 261+24), (0,255,255), 2)
                            cv2.putText(out_img, f"Ore: {anchor['ore']}", (20, 50), 0, 0.7, (0,255,255), 2)
                            cv2.imwrite(f"{OUT}/confirmed/Floor_{f_num:03}_Frame_{anchor['idx']:05}.jpg", out_img)
                            
                            confirmed.append({"floor": f_num, "idx": anchor['idx'], "ore": anchor['ore']})
                            print(f"\n [OK] Committed F{f_num} (Anchor Idx: {anchor['idx']}). New Anchor at Idx: {i}")

                            # UPDATE ANCHOR TO THE NEW STATE
                            anchor = {
                                "num": f_num + 1, "idx": i, "slot": slot, "img": img_bgr.copy(),
                                "hud_bytes": cur_bytes, "hud_visual": cur_roi, "ore": cur_ore
                            }
                            diff_frames_count = 0

    with open("Run_0_FloorMap_v11_4.json", 'w') as f: json.dump(confirmed, f, indent=4)

if __name__ == "__main__":
    run_v11_4_anchor_lock()