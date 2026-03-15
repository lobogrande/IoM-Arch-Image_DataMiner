import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_ROOT = f"diagnostic_results/Leapfrog_v102"
FINAL_DIR = f"{OUTPUT_ROOT}/confirmed_floors"
REJECT_DIR = f"{OUTPUT_ROOT}/rejection_audits"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261
MATCH_THRESHOLD = 0.92

def get_ore_identity(img_gray, slot_idx, templates):
    """Samples the ACTUAL ore slot area and matches against templates."""
    cx = ORE_SLOTS_X[slot_idx]
    # Crop the 44x44 area where the ore actually sits
    roi = img_gray[ORE_Y-22:ORE_Y+22, cx-22:cx+22]
    
    best_name = "none"
    max_val = -1
    for name, tpl in templates.items():
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > max_val:
            max_val = val
            best_name = name
    return best_name if max_val > 0.70 else "empty"

def get_hud_pixels(img_gray):
    """Returns the binarized pixels of the Stage HUD for identity check."""
    roi = img_gray[65:100, 140:200]
    _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    return thresh

def run_v102_leapfrog():
    # 1. Load Templates
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {f.split('.')[0]: cv2.resize(cv2.imread(f"templates/{f}", 0), (44,44)) 
                for f in os.listdir("templates") if f.endswith('.png') and 'player' not in f}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 2. COMMIT FLOOR 1 (The Freebie)
    f1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    
    # pending_floor stores the data for the floor we haven't saved yet
    pending_floor = {
        "floor_num": 1,
        "idx": 0,
        "slot": 0,
        "ore_id": get_ore_identity(f1_gray, 0, ore_tpls),
        "hud_state": get_hud_pixels(f1_gray),
        "img": f1_bgr.copy()
    }
    
    confirmed_library = []

    print(f"--- Running v10.2 Leapfrog Auditor ---")

    for i in range(1, len(buffer_files)):
        if i % 1000 == 0: print(f" [Scanning] Frame {i:05} | Confirmed: {len(confirmed_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # STEP A: FIND A POTENTIAL BOUNDARY (V1.0 Logic)
        search_roi = img_gray[200:350, 0:480]
        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED))

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            anchor_slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(current_x - a) <= 3), None)
            
            if anchor_slot is not None:
                # Shadow check: Left must be empty
                is_clean = True
                for s in range(anchor_slot):
                    if np.mean(img_gray[ORE_Y-5:ORE_Y+5, ORE_SLOTS_X[s]-5:ORE_SLOTS_X[s]+5]) > 60:
                        is_clean = False; break
                
                if is_clean and (i - pending_floor['idx'] > 5):
                    # We found a new boundary candidate.
                    # STEP B: PERFORM COMPETITIVE AUDIT AGAINST THE PENDING FLOOR
                    current_ore = get_ore_identity(img_gray, anchor_slot, ore_tpls)
                    current_hud = get_hud_pixels(img_gray)
                    
                    is_miss_call = False
                    if anchor_slot == pending_floor['slot']:
                        # Compare Ore and HUD
                        ore_match = (current_ore == pending_floor['ore_id'])
                        hud_diff = cv2.absdiff(current_hud, pending_floor['hud_state'])
                        hud_match = (np.sum(hud_diff) < 200) # Threshold for "same number"
                        
                        if ore_match and hud_match:
                            # REJECT: This is just a stationary frame from the same floor
                            is_miss_call = True
                            # Save rejection evidence
                            canvas = np.hstack((cv2.resize(pending_floor['img'], (400,500)), cv2.resize(img_bgr, (400,500))))
                            cv2.putText(canvas, f"REJECTED: Same Ore({current_ore}) & HUD", (20, 40), 0, 0.7, (0,0,255), 2)
                            cv2.imwrite(os.path.join(REJECT_DIR, f"Reject_at_Frame_{i:05}.jpg"), canvas)

                    if not is_miss_call:
                        # SUCCESS: The pending floor is now confirmed.
                        # Save the PREVIOUS floor (Leapfrog)
                        f_num = pending_floor['floor_num']
                        f_idx = pending_floor['idx']
                        
                        # Save Image to confirmed folder
                        out_img = pending_floor['img']
                        cv2.rectangle(out_img, (VALID_ANCHORS[pending_floor['slot']], 225), 
                                      (VALID_ANCHORS[pending_floor['slot']]+40, 285), (0,255,0), 2)
                        cv2.putText(out_img, f"Ore: {pending_floor['ore_id']}", (20, 40), 0, 0.7, (0,255,0), 2)
                        
                        cv2.imwrite(os.path.join(FINAL_DIR, f"Floor_{f_num:03}_Frame_{f_idx:05}.jpg"), out_img)
                        
                        confirmed_library.append({
                            "floor": f_num, "idx": f_idx, "slot": pending_floor['slot'], "ore": pending_floor['ore_id']
                        })
                        
                        print(f"\n [!] CONFIRMED Floor {f_num} | Frame {f_idx}")

                        # Update Pending to the new current frame
                        pending_floor = {
                            "floor_num": f_num + 1,
                            "idx": i,
                            "slot": anchor_slot,
                            "ore_id": current_ore,
                            "hud_state": current_hud,
                            "img": img_bgr.copy()
                        }

    # Save final JSON map
    with open(f"Run_0_FloorMap_v102.json", 'w') as f:
        json.dump(confirmed_library, f, indent=4)

if __name__ == "__main__":
    run_v102_leapfrog()