import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v94_Delta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90
CENSUS_THRESHOLD = 18 # Increased slightly for boss-room stability

def get_grid_census(img_gray):
    states = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74 + (col * 59.1)), int(261 + (row * 59.1))
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        states.append(np.mean(roi))
    return np.array(states)

def run_v94_delta_census():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. FORCE FLOOR 1
    floor_library = [{"floor": 1, "idx": 0}]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    history = []
    last_trigger_idx = 0
    last_clean_census = None
    
    print(f"--- Running v9.4 Delta-Census Auditor ---")

    for i in range(len(buffer_files)):
        if i % 250 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. NOISE ERASER (5-frame Median)
        history.append(img_gray)
        if len(history) > 5: history.pop(0)
        if len(history) < 5: continue
        clean_now = np.median(np.stack(history, axis=0), axis=0).astype(np.uint8)
        clean_now[60:120, :] = 0 # Banner mask

        # 3. PLAYER CHECK
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                
                # 4. THE DELTA CENSUS
                # We compare current CLEAN frame to the PREVIOUS CLEAN frame
                current_census = get_grid_census(clean_now)
                
                if last_clean_census is not None:
                    diffs = np.abs(current_census - last_clean_census)
                    changed_slots = np.sum(diffs > 50)
                    
                    # TRIGGER: If a sudden massive change happened AND we haven't logged recently
                    if changed_slots >= CENSUS_THRESHOLD and (i - last_trigger_idx) > 15:
                        floor_num = len(floor_library) + 1
                        last_trigger_idx = i
                        
                        bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
                        # Restore overlays
                        cv2.putText(bgr_prev, f"F{i-1} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                        cv2.putText(img_bgr, f"F{i} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                        cv2.putText(img_bgr, f"Census: {changed_slots}", (320, 40), 0, 0.6, (255,255,0), 2)
                        
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                    np.hstack((bgr_prev, img_bgr)))
                        
                        floor_library.append({"floor": floor_num, "idx": i})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | Census: {changed_slots}")
                
                last_clean_census = current_census.copy()

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v94_delta_census()