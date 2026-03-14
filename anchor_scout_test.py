import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v7_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE SIX "HOME" ANCHORS (PlayerX values derived from your v6.3/v6.8 logs)
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def run_v7_anchor_snap():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_x = -999
    
    # Save Floor 1
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v7.0 Anchor-Snap Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_gray is None: continue

        # 1. FIND PLAYER COORDINATES
        search_roi = img_gray[230:310, 0:450]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            
            # 2. IS PLAYER AT A "HOME" ANCHOR?
            # Check if current_x is within 2 pixels of any valid starting position
            at_home = any(abs(current_x - a) <= 2 for a in VALID_ANCHORS)
            
            # 3. IS THIS A NEW TELEPORT?
            # Trigger if (At Home) AND (Position changed since last frame)
            if at_home and current_x != last_x:
                # Double-check: Dig Stage text area should be active (light pixels)
                text_roi = img_gray[130:175, 200:380]
                if np.mean(text_roi) > 40: # Simple brightness check for the pulse
                    floor_num = len(floor_library) + 1
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Anchor X: {current_x}")
            
            last_x = current_x

    # Final Map Export
    with open(f"Run_0_FloorMap_v7.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v7_anchor_snap()