import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v87_Identity_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
FLOOR_SWAP_QUIET = 5   # Min bits for a quiet transition
FLOOR_SWAP_SURE = 12   # Bits that guarantee a new floor without HUD check
BANNER_INTENSITY = 248
HUD_CHANGE_CONFIRM = 3.0 # Stricter delta to confirm HUD state reset

def is_banner_present(img_gray):
    corridor = img_gray[200:500, 50:400]
    return np.sum(corridor >= BANNER_INTENSITY) > 35 

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v87_identity_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    active_vector = None
    floor_start_hud = None # THE ANCHOR: HUD state at the absolute start of the floor
    last_valid_idx = 0
    
    print(f"--- Running v8.7 Identity-Locked Audit (Run_{TARGET_RUN}) ---")
    start_time = time.time()

    for i in range(len(buffer_files)):
        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if curr_bgr is None: continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        if is_banner_present(curr_gray): continue

        new_vector = get_existence_vector(curr_gray, bg_t)
        # Surgical ROI: Small box around the 'Stage: XX' number only
        curr_hud_roi = curr_gray[78:105, 115:155] 
        
        if active_vector is None:
            active_vector = new_vector
            floor_start_hud = curr_hud_roi
            last_valid_idx = i
            continue

        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)
        
        # Determine if we have a floor change
        is_new_floor = False
        
        # Case A: Massive layout change (Sure bet)
        if diff_count >= FLOOR_SWAP_SURE:
            is_new_floor = True
        
        # Case B: Quiet change (5-11 bits) + HUD Reset confirm
        elif diff_count >= FLOOR_SWAP_QUIET:
            # Compare current HUD against the START of the floor, not the previous frame
            hud_delta = cv2.absdiff(curr_hud_roi, floor_start_hud).mean()
            if hud_delta > HUD_CHANGE_CONFIRM:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[last_valid_idx]))
            
            cv2.putText(prev_frame, f"END FLOOR {floor_num}", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(curr_bgr, f"START {floor_num+1}", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((prev_frame, curr_bgr)))
            
            floor_library.append({"floor": floor_num, "idx": i, "frame": buffer_files[i]})
            print(f" [!] Boundary {floor_num} Found (Frame {i}) | DNA Delta: {diff_count}")
            
            # LOCK NEW ANCHORS
            active_vector = new_vector
            floor_start_hud = curr_hud_roi
            last_valid_idx = i
            continue

        # Update DNA on mining, but DO NOT update the HUD anchor (keeps it sensitive to fades)
        elif diff_count > 0:
            active_vector = new_vector
            last_valid_idx = i

    with open(os.path.join(OUTPUT_DIR, "v87_identity_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v87_identity_audit()