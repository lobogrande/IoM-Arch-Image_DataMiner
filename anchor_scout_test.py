import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v86_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90
# Noise threshold: How many pixels must change to count as a "shift"
DIFF_THRESHOLD = 1500 

def run_v86_final_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. FORCE FLOOR 1
    floor_library = [{"floor": 1, "idx": 0}]
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)

    last_img_gray = cv2.cvtColor(bgr_start, cv2.COLOR_BGR2GRAY)
    last_trigger_idx = 0
    
    print(f"--- Running v8.6 Difference-First Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_n1_bgr is None: continue
        img_n1_gray = cv2.cvtColor(img_n1_bgr, cv2.COLOR_BGR2GRAY)

        # 2. CALCULATE GLOBAL DIFFERENCE (Noise filter)
        # Focus on the playfield where the teleport/grid shift happens
        diff = cv2.absdiff(img_n1_gray[200:500, :], last_img_gray[200:500, :])
        pixel_change = np.sum(diff > 30) # Count pixels that changed significantly

        # 3. SEEK PLAYER IF CHANGE DETECTED
        if pixel_change > DIFF_THRESHOLD:
            search_roi = img_n1_gray[200:350, 0:480]
            res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > MATCH_THRESHOLD:
                # Is player at a Verified Anchor?
                anchor = next((a for a in VALID_ANCHORS if abs(max_loc[0] - a) <= 4), None)
                
                if anchor is not None and (i - last_trigger_idx) > 10:
                    floor_num = len(floor_library) + 1
                    last_trigger_idx = i
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                    cv2.putText(img_n1_bgr, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, img_n1_bgr)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Change: {pixel_change}")

        last_img_gray = img_n1_gray

    with open(f"Run_0_FloorMap_v86.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v86_final_audit()