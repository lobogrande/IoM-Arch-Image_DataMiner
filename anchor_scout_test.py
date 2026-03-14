import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v73_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE SIX VERIFIED ANCHORS (from your Truth screenshots)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def run_v73_proximity_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_x = -999
    last_trigger_idx = -20 # Cooldown to prevent double-triggers

    # Save Floor 1 Force
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v7.3 Proximity Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors Found: {len(floor_library)}", end='\r')

        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_gray is None: continue

        # 1. FIND PLAYER (Using the same Y-band from Ground Truth v7.2)
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0] # This is absolute X since search_roi starts at X=0
            
            # 2. IS PLAYER AT A "HOME" ANCHOR?
            current_anchor = None
            for a in VALID_ANCHORS:
                if abs(current_x - a) <= 4: # 4px tolerance
                    current_anchor = a
                    break
            
            # 3. TRIGGER LOGIC: 
            # If we are at an anchor AND (we weren't there last frame OR we were at a DIFFERENT anchor)
            if current_anchor is not None:
                if (current_x != last_x) and (i - last_trigger_idx > 10):
                    floor_num = len(floor_library) + 1
                    last_trigger_idx = i
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Anchor: {current_anchor}")
            
            last_x = current_x

    # Export map
    with open(f"Run_0_FloorMap_v73.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v73_proximity_audit()