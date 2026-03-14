import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v61_Fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS
MATCH_THRESHOLD = 0.88  
OFFSET_X = 24           
D_GATE_LIVE = 6.5       

def get_slot_state(roi, bg_template):
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    return 2 if diff > D_GATE_LIVE else 0

def run_v61_fast_audit():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # FORCED START (Floor 1)
    floor_library = [{"floor": 1, "idx": 0}]
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.putText(bgr_start, "DATASET START - FLOOR 1", (30, 50), 0, 0.7, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)
    
    last_logged_dna = None

    print(f"--- Running v6.1 Fast Elastic Auditor ---")
    print(f"Total Frames: {len(buffer_files)}")

    for i in range(len(buffer_files) - 1):
        # HEARTBEAT
        if i % 100 == 0:
            print(f" [Heartbeat] Processing Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. OPTIMIZED SEARCH: Narrow Y-band (240-300) to speed up matching
        search_roi = img_n1_gray[240:300, 0:420] 
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. RELATIVE CALCULATION
            # Adjust max_loc to screen coordinates (search_roi starts at Y=240)
            player_center_x = max_loc[0] + 20
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if 0 <= target_col < 6:
                # 3. DNA ROW SCAN
                current_row_dna = []
                for c in range(6):
                    cx, cy = int(74 + (c * 59.1)), 261
                    roi = img_n1_gray[cy-5:cy+5, cx-5:cx+5]
                    current_row_dna.append(get_slot_state(roi, bg_t[0]))

                # 4. FIRST-LIVE CHECK
                if current_row_dna[target_col] == 2:
                    # Logic: No ores/shadows to the left of the player
                    if all(state == 0 for state in current_row_dna[:target_col]):
                        
                        # 5. PERSISTENCE (Only log if Row 1 has actually changed)
                        if current_row_dna != last_logged_dna:
                            floor_num = len(floor_library) + 1
                            last_logged_dna = current_row_dna
                            
                            # Log Transition Image
                            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                            
                            floor_library.append({"floor": floor_num, "idx": i+1})
                            print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Col {target_col}")

    # Final Summary JSON
    with open(f"Run_0_FloorMap_v61.json", 'w') as f:
        json.dump(floor_library, f)

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v61_fast_audit()