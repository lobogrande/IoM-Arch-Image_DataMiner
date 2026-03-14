import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v71_GPS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE SIX "HOME" ANCHORS
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def run_v71_gps_auditor():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_logged_floor_idx = -100 # Prevent double-logging the same teleport

    # Save Floor 1
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v7.1 Visual GPS Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Found: {len(floor_library)}", end='\r')

        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_gray is None: continue

        # 1. FIND PLAYER
        search_roi = img_gray[230:310, 0:450]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            
            # 2. PROXIMITY CHECK (Is the player within 5px of a Home Anchor?)
            found_anchor = None
            for a in VALID_ANCHORS:
                if abs(current_x - a) <= 5:
                    found_anchor = a
                    break
            
            # 3. PERMISSION GATE
            if found_anchor is not None:
                # Is the "Dig Stage" text pulse active?
                text_roi = img_gray[130:175, 200:380]
                text_active = np.mean(text_roi) > 35 
                
                # Only log if we haven't logged a floor in the last 5 frames
                if text_active and (i - last_logged_floor_idx) > 5:
                    floor_num = len(floor_library) + 1
                    last_logged_floor_idx = i
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Locked to Anchor: {found_anchor} (Real X: {current_x})")
            
            # DIAGNOSTIC: If score is perfect but no anchor found
            elif max_val > 0.93:
                 # Uncomment the line below if it's still silent to see the "Off-Grid" X values
                 # print(f" [Debug] Strong Match at Frame {i+1} but X={current_x} is not near an anchor.")
                 pass

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v71_gps_auditor()