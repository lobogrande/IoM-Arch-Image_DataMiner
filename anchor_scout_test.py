import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v81_SteelTrap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92 # High precision

def run_v81_steel_trap():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_x = -999
    last_logged_stage_roi = None # Memory of the Stage Number pixels

    # Forced Start
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v8.1 Steel-Trap Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_n1_bgr is None: continue
        img_n1_gray = cv2.cvtColor(img_n1_bgr, cv2.COLOR_BGR2GRAY)

        # 1. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        current_at_anchor = False
        current_x = -999
        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 3 for a in VALID_ANCHORS):
                current_at_anchor = True

        # 2. TRIGGER: Teleport detected (Snap from field to anchor)
        if current_at_anchor and (current_x != last_x):
            # 3. THE STEEL TRAP: Has the Stage Number actually changed?
            # ROI for the "Stage: XXX" number box
            stage_roi = img_n1_gray[65:105, 110:190] 
            
            # Binary comparison of the pixels
            if last_logged_stage_roi is None or not np.array_equal(stage_roi, last_logged_stage_roi):
                floor_num = len(floor_library) + 1
                last_logged_stage_roi = stage_roi.copy()
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                
                # Visual Labels
                cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                cv2.putText(img_n1_bgr, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, img_n1_bgr)))
                floor_library.append({"floor": floor_num, "idx": i+1})
                
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Stage Updated")

        last_x = current_x

    # Save final map
    with open(f"Run_0_FloorMap_v81.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v81_steel_trap()