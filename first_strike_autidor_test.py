import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_ROOT = f"diagnostic_results/Leapfrog_v103"
FINAL_DIR = f"{OUTPUT_ROOT}/confirmed_floors"
REJECT_DIR = f"{OUTPUT_ROOT}/rejection_audits"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261
MATCH_THRESHOLD = 0.92

def get_hud_pixels(img_gray):
    """Isolates the 'Stage: XXX' number for identity checks."""
    # Specific ROI for the numeric part of the Stage HUD
    roi = img_gray[65:105, 135:200]
    _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    return thresh

def get_ore_identity(img_gray, slot_idx, templates):
    """Samples ONLY the grid slot, never the player area."""
    cx = ORE_SLOTS_X[slot_idx]
    # Crop a 40x40 area centered on the ORE, not the player
    roi = img_gray[ORE_Y-20:ORE_Y+20, cx-20:cx+20]
    
    best_name = "none"
    max_val = -1
    for name, tpl in templates.items():
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > max_val:
            max_val = val
            best_name = name
    return best_name if max_val > 0.65 else "empty"

def run_v103_competitive_audit():
    # 1. LOAD TEMPLATES
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {f.split('.')[0]: cv2.resize(cv2.imread(f"templates/{f}", 0), (40,40)) 
                for f in os.listdir("templates") if f.endswith('.png') and 'player' not in f}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 2. SEED INITIAL CANDIDATE (Floor 1)
    f1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    
    pending = {
        "num": 1, "idx": 0, "slot": 0, "img": f1_bgr.copy(),
        "hud": get_hud_pixels(f1_gray), "ore": get_ore_identity(f1_gray, 0, ore_tpls)
    }
    
    confirmed_map = []
    print(f"--- Running v10.3 Competitive Auditor ---")

    for i in range(1, len(buffer_files)):
        if i % 1000 == 0: print(f" [Scan] {i:05} | Confirmed: {len(confirmed_map)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # STEP A: PLAYER DETECTION (V1.0 Logic)
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 3), None)
            
            if slot is not None:
                # Shadow check: Is the row empty to the left?
                is_spawn = True
                for s in range(slot):
                    if np.mean(img_gray[ORE_Y-5:ORE_Y+5, ORE_SLOTS_X[s]-5:ORE_SLOTS_X[s]+5]) > 60:
                        is_spawn = False; break
                
                if is_spawn and (i - pending['idx'] > 5):
                    # We have a candidate boundary.
                    cur_hud = get_hud_pixels(img_gray)
                    cur_ore = get_ore_identity(img_gray, slot, ore_tpls)
                    
                    is_miss_call = False
                    if slot == pending['slot']:
                        # THE COMPETITIVE AUDIT: HUD and ORE must differ
                        hud_diff = np.sum(cv2.absdiff(cur_hud, pending['hud']))
                        ore_match = (cur_ore == pending['ore'])
                        
                        if hud_diff < 150 and ore_match:
                            is_miss_call = True
                            # Save Rejection Evidence
                            canvas = np.hstack((cv2.resize(pending['img'], (400,500)), cv2.resize(img_bgr, (400,500))))
                            cv2.putText(canvas, f"REJECT: Same HUD/Ore({cur_ore})", (20, 40), 0, 0.6, (0,0,255), 2)
                            cv2.imwrite(os.path.join(REJECT_DIR, f"Reject_Frame_{i:05}.jpg"), canvas)

                    if not is_miss_call:
                        # LEAPFROG COMMIT: Save the pending floor
                        f_num = pending['num']
                        out_img = pending['img']
                        # Draw box on ORE slot, not player
                        ox = ORE_SLOTS_X[pending['slot']]
                        cv2.rectangle(out_img, (ox-22, ORE_Y-22), (ox+22, ORE_Y+22), (255, 255, 0), 2)
                        cv2.putText(out_img, f"Ore: {pending['ore']}", (ox-20, ORE_Y-30), 0, 0.5, (255,255,0), 1)
                        
                        cv2.imwrite(os.path.join(FINAL_DIR, f"Floor_{f_num:03}_Frame_{pending['idx']:05}.jpg"), out_img)
                        confirmed_map.append({"floor": f_num, "idx": pending['idx'], "ore": pending['ore']})
                        
                        print(f"\n [!] CONFIRMED F{f_num} at Frame {pending['idx']}")

                        # Leapfrog: Current candidate becomes the new pending
                        pending = {
                            "num": f_num + 1, "idx": i, "slot": slot, 
                            "img": img_bgr.copy(), "hud": cur_hud, "ore": cur_ore
                        }

    with open(f"Run_0_FloorMap_v103.json", 'w') as f:
        json.dump(confirmed_map, f, indent=4)

if __name__ == "__main__":
    run_v103_competitive_audit()