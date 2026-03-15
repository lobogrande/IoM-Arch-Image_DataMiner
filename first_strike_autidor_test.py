import cv2
import numpy as np
import os
import json

# --- 1. LEGACY COORDINATES & CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X = 59.1
AI_DIM = 48
# ROIs from your legacy snippets
HEADER_ROI = (54, 74, 103, 138)  # y1, y2, x1, x2
DIG_VAL_ROI = (230, 246, 250, 281)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def get_bitwise_ocr(roi, digit_map, thresh_val):
    """Legacy bitwise OCR logic."""
    _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    found_digits = []
    for x in range(0, thresh.shape[1] - 8):
        slice_roi = thresh[:, x:x+12]
        if np.sum(slice_roi) < 30: continue
        best_v, best_d = -1, None
        for d, tpls in digit_map.items():
            for t_img in tpls:
                # Resize template to match ROI height if necessary
                if t_img.shape[0] != slice_roi.shape[0]:
                    t_img = cv2.resize(t_img, (t_img.shape[1], slice_roi.shape[0]))
                res = cv2.matchTemplate(slice_roi, t_img, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > 0.82 and val > best_v:
                    best_v, best_d = val, d
        if best_d is not None: found_digits.append(str(best_d))
    
    res = ""
    if found_digits:
        res = found_digits[0]
        for i in range(1, len(found_digits)):
            if found_digits[i] != found_digits[i-1]: res += found_digits[i]
    return res

def get_ore_consensus(img_gray, slot_idx, templates, mask):
    """Legacy Masked Ore Matching with Background Competition."""
    cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
    cy = SLOT1_CENTER[1]
    x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
    roi = img_gray[y1:y2, x1:x2]
    
    if roi.shape != (48, 48): roi = cv2.resize(roi, (48, 48))

    best_bg_score = 0.0
    for bg_img in templates['bg']:
        bg_res = cv2.matchTemplate(roi, bg_img, cv2.TM_CCOEFF_NORMED).max()
        if bg_res > best_bg_score: best_bg_score = bg_res

    best_ore = {'tier': 'empty', 'score': 0.0}
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for t_img in types[state]:
                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCORR_NORMED, mask=mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_ore['score']:
                    best_ore = {'tier': tier, 'score': score}

    # Legacy Gate: Score > 0.80 AND outscores BG by 0.06
    if best_ore['score'] > 0.80 and (best_ore['score'] - best_bg_score > 0.06):
        return best_ore['tier']
    return "empty"

def run_v11_audit():
    # Load Assets
    mask = get_spatial_mask()
    player_t = cv2.imread("templates/player_right.png", 0)
    
    # Load Ores (Legacy Format)
    ore_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(f"templates/{f}", 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): ore_tpls['bg'].append(img)
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    # Load Digits
    digit_map = {i: [] for i in range(10)}
    for f in os.listdir("digits"):
        if f.endswith('.png'):
            val = int(f[0])
            digit_map[val].append(cv2.imread(f"digits/{f}", 0))

    BUFFER_ROOT = "capture_buffer_0"
    OUT = "diagnostic_results/LegacyForensic_v11"; os.makedirs(f"{OUT}/confirmed", exist_ok=True); os.makedirs(f"{OUT}/rejects", exist_ok=True)
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    # Seed Floor 1
    f1_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)
    pending = {
        "num": 1, "idx": 0, "slot": 0, "img": cv2.imread(os.path.join(BUFFER_ROOT, files[0])),
        "stage": get_bitwise_ocr(f1_gray[54:74, 103:138], digit_map, 175),
        "ore": get_ore_consensus(f1_gray, 0, ore_tpls, mask)
    }

    confirmed = []
    print("--- Running v11.0 Legacy-Hardened Auditor ---")

    for i in range(1, len(files)):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Candidate Check
        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(img_gray[200:350, 0:480], player_t, cv2.TM_CCOEFF_NORMED))
        if max_val > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 3), None)
            if slot is not None:
                # Shadow Check (Row 1 Only)
                is_spawn = True
                for s in range(slot):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    if np.mean(img_gray[261-5:261+5, cx-5:cx+5]) > 60: is_spawn = False; break
                
                if is_spawn and (i - pending['idx'] > 4):
                    cur_h = get_bitwise_ocr(img_gray[54:74, 103:138], digit_map, 175)
                    cur_o = get_ore_consensus(img_gray, slot, ore_tpls, mask)
                    
                    # REJECTION GATE
                    if slot == pending['slot'] and cur_h == pending['stage'] and cur_h != "":
                        if cur_o == "empty" or cur_o == pending['ore']:
                            canvas = np.hstack((cv2.resize(pending['img'], (400,500)), cv2.resize(img_bgr, (400,500))))
                            cv2.putText(canvas, f"REJECT: S{cur_h} {cur_o}", (10, 40), 0, 0.7, (0,0,255), 2)
                            cv2.imwrite(f"{OUT}/rejects/Reject_S{cur_h}_F{i}.jpg", canvas)
                            continue

                    # CONFIRM LEAPFROG
                    f_num = pending['num']
                    out_img = pending['img']
                    cx = int(SLOT1_CENTER[0] + (pending['slot'] * STEP_X))
                    cv2.rectangle(out_img, (cx-24, 261-24), (cx+24, 261+24), (0,255,255), 2)
                    cv2.putText(out_img, f"Ore:{pending['ore']} S:{pending['stage']}", (20, 50), 0, 0.7, (0,255,255), 2)
                    cv2.imwrite(f"{OUT}/confirmed/Floor_{f_num:03}_Frame_{pending['idx']:05}.jpg", out_img)
                    confirmed.append({"floor": f_num, "idx": pending['idx'], "stage": pending['stage']})
                    print(f" [OK] F{f_num} | S:{pending['stage']} | Frame:{pending['idx']}")

                    pending = {"num": f_num+1, "idx": i, "slot": slot, "img": img_bgr.copy(), "stage": cur_h, "ore": cur_o}

    with open("Run_0_FloorMap_v11.json", 'w') as f: json.dump(confirmed, f, indent=4)

if __name__ == "__main__":
    run_v11_audit()