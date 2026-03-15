import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v92_Forensic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def run_v92_forensic_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. FORCE FLOOR 1 IMMEDIATELY
    floor_library = [{"floor": 1, "idx": 0}]
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.putText(bgr_start, "DATASET START (F1)", (20, 40), 0, 0.7, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)

    history = []
    last_trigger_idx = 0
    last_clean_grid = None
    
    print(f"--- Running v9.2 Forensic Auditor ---")

    for i in range(len(buffer_files)):
        if i % 100 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. MAINTAIN 3-FRAME HISTORY (Median Eraser)
        history.append(img_gray)
        if len(history) > 3: history.pop(0)
        if len(history) < 3: continue

        # Generate "Clean" version of the current frame
        clean_now = np.median(np.stack(history, axis=0), axis=0).astype(np.uint8)

        # 3. SEEK PLAYER (Original frame for max confidence)
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            # Is player at a verified anchor?
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                
                # 4. VERIFY GRID RESET
                # We compare the cleaned grid now vs. the cleaned grid from the last trigger
                if last_clean_grid is not None:
                    diff = cv2.absdiff(clean_now[200:450, :], last_clean_grid[200:450, :])
                    # Standard Deviation is high when the whole screen changes
                    # Low when only small numbers/fairies move
                    change_score = np.std(diff)
                    
                    if change_score > 12 and (i - last_trigger_idx) > 10:
                        floor_num = len(floor_library) + 1
                        last_trigger_idx = i
                        last_clean_grid = clean_now.copy()
                        
                        bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
                        cv2.putText(bgr_prev, f"F{i-1} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                        cv2.putText(img_bgr, f"F{i} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                        
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                    np.hstack((bgr_prev, img_bgr)))
                        
                        floor_library.append({"floor": floor_num, "idx": i})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | STD: {change_score:.2f}")
                else:
                    # Initialize the first grid
                    last_clean_grid = clean_now.copy()
                    last_trigger_idx = i

    # Export map
    with open(f"Run_0_FloorMap_v92.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v92_forensic_audit()