import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v93_Census"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90
CENSUS_THRESHOLD = 16 # Number of slots that must change to count as a new floor

def get_grid_census(img_gray):
    """Checks the state of all 24 ore slots."""
    states = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74 + (col * 59.1)), int(261 + (row * 59.1))
        # Small 10x10 sample of the slot center
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        states.append(np.mean(roi))
    return np.array(states)

def run_v93_census_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # Force Floor 1
    floor_library = [{"floor": 1, "idx": 0}]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    history = []
    last_trigger_idx = 0
    last_census_states = None
    
    print(f"--- Running v9.3 Census Auditor ---")

    for i in range(len(buffer_files)):
        if i % 250 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. NOISE ERASER (3-frame Median)
        history.append(img_gray)
        if len(history) > 3: history.pop(0)
        if len(history) < 3: continue
        clean_now = np.median(np.stack(history, axis=0), axis=0).astype(np.uint8)

        # 2. BANNER MASKING
        # Black out the banner zone so it doesn't affect the census
        clean_now[60:120, :] = 0

        # 3. PLAYER CHECK
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                
                # 4. THE CENSUS
                current_census = get_grid_census(clean_now)
                
                if last_census_states is not None:
                    # How many slots changed by more than 40 brightness units?
                    diffs = np.abs(current_census - last_census_states)
                    changed_slots = np.sum(diffs > 40)
                    
                    if changed_slots >= CENSUS_THRESHOLD and (i - last_trigger_idx) > 10:
                        floor_num = len(floor_library) + 1
                        last_trigger_idx = i
                        last_census_states = current_census.copy()
                        
                        bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
                        cv2.putText(img_bgr, f"Census: {changed_slots}", (300, 40), 0, 0.7, (255,255,0), 2)
                        
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                    np.hstack((bgr_prev, img_bgr)))
                        
                        floor_library.append({"floor": floor_num, "idx": i})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | Census: {changed_slots}")
                else:
                    last_census_states = current_census.copy()
                    last_trigger_idx = i

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v93_census_audit()