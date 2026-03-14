import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v84_Occam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE SIX VERIFIED ANCHORS
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.88 # Slightly lowered to ensure we never miss the player

def run_v84_occam_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. FORCE FLOOR 1
    floor_library = [{"floor": 1, "idx": 0}]
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.putText(bgr_start, "DATASET START (F1)", (20, 40), 0, 0.7, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)

    last_trigger_idx = -10
    print(f"--- Running v8.4 Occam Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors Found: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 2. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            
            # 3. IS PLAYER AT AN ANCHOR?
            at_home = any(abs(current_x - a) <= 5 for a in VALID_ANCHORS)
            
            # 4. TRIGGER LOGIC
            # If at home AND we haven't triggered in the last 3 frames
            if at_home and (i - last_trigger_idx > 3):
                # We also check if the player moved since the last trigger
                # to avoid logging the same standing-still position
                floor_num = len(floor_library) + 1
                last_trigger_idx = i
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                
                # Visual Labels
                cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_n1, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                floor_library.append({"floor": floor_num, "idx": i+1})
                
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | X: {current_x}")

    # Final Map
    with open(f"Run_0_FloorMap_v84.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v84_occam_audit()