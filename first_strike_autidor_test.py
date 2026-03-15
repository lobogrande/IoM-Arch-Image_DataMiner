import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_ROOT = f"diagnostic_results/Leapfrog_v107"
FINAL_DIR = f"{OUTPUT_ROOT}/confirmed_floors"
REJECT_DIR = f"{OUTPUT_ROOT}/rejection_audits"

for d in [FINAL_DIR, REJECT_DIR]: os.makedirs(d, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261
MATCH_THRESHOLD = 0.92

def get_stage_number(img_gray, digit_tpls):
    """OCR that checks against all variants (dim, bright, noisy) for digits 0-9."""
    roi = img_gray[65:105, 140:210] 
    _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    
    found_digits = []
    # Horizontal scan
    for x in range(0, thresh.shape[1]-10):
        slice_roi = thresh[:, x:x+12]
        if np.sum(slice_roi) < 40: continue
        
        best_v, best_d = -1, None
        # Check slice against all variants of all digits
        for digit_val, variants in digit_tpls.items():
            for v_img in variants:
                res = cv2.matchTemplate(slice_roi, v_img, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > 0.86 and val > best_v:
                    best_v, best_d = val, digit_val
        
        if best_d is not None:
            found_digits.append(str(best_d))
    
    # Deduplicate and Join
    res_str = ""
    if found_digits:
        res_str = found_digits[0]
        for i in range(1, len(found_digits)):
            if found_digits[i] != found_digits[i-1]:
                res_str += found_digits[i]
    return res_str

def get_ore_prefix(img_gray, slot_idx, templates):
    cx = ORE_SLOTS_X[slot_idx]
    roi = img_gray[ORE_Y-22:ORE_Y+22, cx-22:cx+22]
    best_p, max_v = "empty", -1
    for name, tpl in templates.items():
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > 0.68 and val > max_v:
            max_v, best_p = val, name.split('_')[0]
    return best_p

def run_v107_audit():
    # 1. LOAD TEMPLATES
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {f.split('.')[0]: cv2.resize(cv2.imread(f"templates/{f}", 0), (44,44)) 
                for f in os.listdir("templates") if f.endswith('.png') and '_' in f}
    
    # 2. LOAD DIGIT VARIANTS (digits/ folder)
    digit_tpls = {i: [] for i in range(10)}
    for f in os.listdir("digits"):
        if f.endswith('.png'):
            d_val = int(f[0])
            tpl = cv2.resize(cv2.imread(f"digits/{f}", 0), (12, 20))
            digit_tpls[d_val].append(tpl)
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 3. SEED INITIAL
    f1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    pending = {
        "num": 1, "idx": 0, "slot": 0, "img": f1_bgr.copy(),
        "stage": get_stage_number(f1_gray, digit_tpls),
        "prefix": get_ore_prefix(f1_gray, 0, ore_tpls)
    }
    
    confirmed_map = []
    print(f"--- Running v10.7 Multi-Variant Auditor ---")

    for i in range(1, len(buffer_files)):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Boundary Candidate Logic
        search_roi = img_gray[200:350, 0:480]
        _, max_v, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED))

        if max_v > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 3), None)
            if slot is not None:
                # Shadow Check
                is_spawn = True
                for s in range(slot):
                    if np.mean(img_gray[ORE_Y-5:ORE_Y+5, ORE_SLOTS_X[s]-5:ORE_SLOTS_X[s]+5]) > 60:
                        is_spawn = False; break
                
                if is_spawn and (i - pending['idx'] > 3):
                    cur_stage = get_stage_number(img_gray, digit_tpls)
                    cur_prefix = get_ore_prefix(img_gray, slot, ore_tpls)
                    
                    is_miss = False
                    # COMPETITIVE AUDIT: Same Slot + Same Stage String + Same Prefix
                    if slot == pending['slot'] and cur_stage == pending['stage'] and cur_stage != "":
                        # Only use ore match if it's not 'empty'
                        if cur_prefix == "empty" or cur_prefix == pending['prefix']:
                            is_miss = True
                            # Evidence stack
                            canvas = np.hstack((cv2.resize(pending['img'], (400,500)), cv2.resize(img_bgr, (400,500))))
                            cv2.putText(canvas, f"REJECT: S{cur_stage} Ore:{cur_prefix}", (10, 40), 0, 0.7, (0,0,255), 2)
                            cv2.imwrite(os.path.join(REJECT_DIR, f"Reject_S{cur_stage}_F{i}.jpg"), canvas)
                    
                    if not is_miss:
                        # CONFIRM PENDING
                        f_num = pending['num']
                        out_img = pending['img']
                        ox = ORE_SLOTS_X[pending['slot']]
                        cv2.rectangle(out_img, (ox-22, ORE_Y-22), (ox+22, ORE_Y+22), (0, 255, 255), 2)
                        cv2.putText(out_img, f"Ore: {pending['prefix']} | S: {pending['stage']}", (20, 50), 0, 0.7, (0,255,255), 2)
                        
                        cv2.imwrite(os.path.join(FINAL_DIR, f"Floor_{f_num:03}_Frame_{pending['idx']:05}.jpg"), out_img)
                        confirmed_map.append({"floor": f_num, "idx": pending['idx'], "stage": pending['stage']})
                        print(f" [OK] Confirmed F{f_num} | S:{pending['stage']} | Frame {pending['idx']}")

                        # Leapfrog
                        pending = {"num": f_num+1, "idx": i, "slot": slot, "img": img_bgr.copy(), "stage": cur_stage, "prefix": cur_prefix}

    with open(f"Run_0_FloorMap_v107.json", 'w') as f:
        json.dump(confirmed_map, f, indent=4)

if __name__ == "__main__":
    run_v107_audit()