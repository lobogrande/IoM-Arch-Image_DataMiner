import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_ROOT = f"diagnostic_results/Forensic_v104"
FINAL_DIR = f"{OUTPUT_ROOT}/confirmed_floors"
REJECT_DIR = f"{OUTPUT_ROOT}/rejection_audits"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261
MATCH_THRESHOLD = 0.92

def get_stage_number(img_gray, digit_tpls):
    """Uses template matching to read the Stage Number as an integer."""
    roi = img_gray[65:105, 140:200]
    _, thresh = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)
    
    found_digits = []
    # Simplified horizontal scan for digits
    for x in range(0, thresh.shape[1]-10):
        slice_roi = thresh[:, x:x+12]
        if np.sum(slice_roi) < 50: continue
        
        best_val = -1
        best_digit = None
        for d, tpl in digit_tpls.items():
            res = cv2.matchTemplate(slice_roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > 0.85 and val > best_val:
                best_val = val
                best_digit = d
        if best_digit is not None:
            found_digits.append(str(best_digit))
    
    return "".join(dict.fromkeys(found_digits)) # Remove rapid duplicates

def get_ore_prefix(img_gray, slot_idx, templates):
    """Returns the prefix (e.g., 'epic2') of the ore in the slot."""
    cx = ORE_SLOTS_X[slot_idx]
    roi = img_gray[ORE_Y-22:ORE_Y+22, cx-22:cx+22]
    
    best_name = "none"
    max_val = -1
    for name, tpl in templates.items():
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > max_val:
            max_val = val
            best_name = name.split('_')[0] # Take prefix before first underscore
            
    return best_name if max_val > 0.65 else "empty"

def run_v104_forensic():
    # 1. LOAD ALL TEMPLATES
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {f.split('.')[0]: cv2.resize(cv2.imread(f"templates/{f}", 0), (44,44)) 
                for f in os.listdir("templates") if f.endswith('.png') and 'player' not in f and not f[0].isdigit()}
    digit_tpls = {int(f.split('.')[0]): cv2.resize(cv2.imread(f"templates/{f}", 0), (12,20))
                  for f in os.listdir("templates") if f.endswith('.png') and f[0].isdigit()}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 2. SEED FLOOR 1
    f1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    
    pending = {
        "num": 1, "idx": 0, "slot": 0, "img": f1_bgr.copy(),
        "stage": get_stage_number(f1_gray, digit_tpls),
        "prefix": get_ore_prefix(f1_gray, 0, ore_tpls)
    }
    
    confirmed_map = []
    print(f"--- Running v10.4 Forensic Auditor ---")

    for i in range(1, len(buffer_files)):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # FIND CANDIDATE
        search_roi = img_gray[200:350, 0:480]
        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED))

        if max_val > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 3), None)
            
            if slot is not None:
                # Shadow check
                is_spawn = True
                for s in range(slot):
                    if np.mean(img_gray[ORE_Y-5:ORE_Y+5, ORE_SLOTS_X[s]-5:ORE_SLOTS_X[s]+5]) > 60:
                        is_spawn = False; break
                
                if is_spawn and (i - pending['idx'] > 5):
                    # NEW CANDIDATE FOUND
                    cur_stage = get_stage_number(img_gray, digit_tpls)
                    cur_prefix = get_ore_prefix(img_gray, slot, ore_tpls)
                    
                    is_miss_call = False
                    if slot == pending['slot']:
                        # COMPETITIVE AUDIT
                        stage_match = (cur_stage == pending['stage']) and (cur_stage != "")
                        prefix_match = (cur_prefix == pending['prefix']) and (cur_prefix != "empty")
                        
                        if stage_match and prefix_match:
                            is_miss_call = True
                            # SAVE REJECTION EVIDENCE
                            canvas = np.hstack((cv2.resize(pending['img'], (400,500)), cv2.resize(img_bgr, (400,500))))
                            cv2.putText(canvas, f"REJECT: Stage {cur_stage} | Ore {cur_prefix}", (10, 30), 0, 0.6, (0,0,255), 2)
                            cv2.imwrite(os.path.join(REJECT_DIR, f"Reject_Stage{cur_stage}_Frame{i}.jpg"), canvas)

                    if not is_miss_call:
                        # CONFIRM PREVIOUS
                        f_num = pending['num']
                        out_img = pending['img']
                        ox = ORE_SLOTS_X[pending['slot']]
                        cv2.rectangle(out_img, (ox-22, ORE_Y-22), (ox+22, ORE_Y+22), (0, 255, 255), 2)
                        cv2.putText(out_img, f"Ore: {pending['prefix']} | S:{pending['stage']}", (20, 40), 0, 0.7, (0,255,255), 2)
                        
                        cv2.imwrite(os.path.join(FINAL_DIR, f"Floor_{f_num:03}_Frame_{pending['idx']:05}.jpg"), out_img)
                        confirmed_map.append({"floor": f_num, "idx": pending['idx'], "stage": pending['stage'], "ore": pending['prefix']})
                        print(f"\n [!] CONFIRMED F{f_num} | Stage {pending['stage']} | Frame {pending['idx']}")

                        # LEAPFROG
                        pending = {
                            "num": f_num + 1, "idx": i, "slot": slot, 
                            "img": img_bgr.copy(), "stage": cur_stage, "prefix": cur_prefix
                        }

    with open(f"Run_0_FloorMap_v104.json", 'w') as f:
        json.dump(confirmed_map, f, indent=4)

if __name__ == "__main__":
    run_v104_forensic()