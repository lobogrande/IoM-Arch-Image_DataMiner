import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FirstStrike_v101"
REJECT_DIR = f"{OUTPUT_DIR}/rejection_audits"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261
MATCH_THRESHOLD = 0.92

def get_ore_id(img_gray, slot_idx, templates):
    """Identifies the ore in the slot using template matching."""
    cx = ORE_SLOTS_X[slot_idx]
    # Crop the specific slot area (slightly tight to avoid floor noise)
    roi = img_gray[ORE_Y-20:ORE_Y+24, cx-24:cx+24]
    
    best_match = "unknown"
    max_val = -1
    
    for name, tpl in templates.items():
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > max_val:
            max_val = val
            best_match = name
    
    return best_match if max_val > 0.65 else "empty"

def get_hud_hash(img_gray):
    """Hashes the Stage Number box for identical-state detection."""
    roi = img_gray[65:100, 140:200]
    return hash(roi.tobytes())

def run_v101_audit():
    # Load Templates
    player_t = cv2.imread("templates/player_right.png", 0)
    ore_tpls = {f.split('.')[0]: cv2.resize(cv2.imread(f"templates/{f}", 0), (40,40)) 
                for f in os.listdir("templates") if f.endswith('.png') and 'player' not in f}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # Init Floor 1
    f1_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    floor_library = [{"floor": 1, "idx": 0, "slot": 0, "ore": "start", "hash": get_hud_hash(cv2.cvtColor(f1_img, cv2.COLOR_BGR2GRAY))}]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001_F00000.jpg"), f1_img)

    print(f"--- Running v10.1 Double-Pass Auditor ---")

    for i in range(1, len(buffer_files)):
        if i % 1000 == 0: print(f" [Scanning] Frame {i} | Found: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # PASS 1: Player Position & Shadow Check
        search_roi = img_gray[200:350, 0:480]
        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED))

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            anchor_slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(current_x - a) <= 3), None)
            
            if anchor_slot is not None:
                # Shadow check: No ores to the left
                is_clean_spawn = True
                for s in range(anchor_slot):
                    cx = ORE_SLOTS_X[s]
                    if np.mean(img_gray[ORE_Y-5:ORE_Y+5, cx-5:cx+5]) > 60:
                        is_clean_spawn = False; break
                
                if is_clean_spawn:
                    # PASS 2: Double-Verification
                    current_ore = get_ore_id(img_gray, anchor_slot, ore_tpls)
                    current_hash = get_hud_hash(img_gray)
                    last_call = floor_library[-1]

                    # If player at same slot, verify Ore & HUD
                    if anchor_slot == last_call['slot']:
                        if current_ore == last_call['ore'] and current_hash == last_call['hash']:
                            # REJECTED: Stationary Miss-call
                            bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[last_call['idx']]))
                            # Annotation
                            cv2.putText(img_bgr, f"REJECT: Same Ore ({current_ore})", (20, 50), 0, 0.6, (0,0,255), 2)
                            comparison = np.hstack((cv2.resize(bgr_prev, (480, 640)), cv2.resize(img_bgr, (480, 640))))
                            cv2.imwrite(os.path.join(REJECT_DIR, f"Reject_F{len(floor_library)+1}_at_Frame_{i}.jpg"), comparison)
                            continue

                    # Confirmed New Floor
                    floor_num = len(floor_library) + 1
                    floor_library.append({"floor": floor_num, "idx": i, "slot": anchor_slot, "ore": current_ore, "hash": current_hash})
                    
                    cv2.rectangle(img_bgr, (max_loc[0], max_loc[1]+200), (max_loc[0]+40, max_loc[1]+260), (0,255,0), 2)
                    cv2.putText(img_bgr, f"Ore: {current_ore}", (max_loc[0], max_loc[1]+190), 0, 0.5, (0,255,0), 1)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}_Frame_{i:05}.jpg"), img_bgr)
                    print(f"\n [!] CONFIRMED: Floor {floor_num} | Frame {i} | Ore: {current_ore}")

    with open(f"Run_0_FloorMap_v101.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v101_audit()