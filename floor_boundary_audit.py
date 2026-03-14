import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v18_Surgical_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.0         

def get_surgical_vector(img_gray, bg_templates):
    """Triple-ROI consensus per slot to ignore damage-number noise."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # 10x10 Surgical zones: Top, Center, Bottom
        zones = [img_gray[cy-15:cy-5, cx-5:cx+5], img_gray[cy-5:cy+5, cx-5:cx+5], img_gray[cy+5:cy+15, cx-5:cx+5]]
        votes = 0
        for z in zones:
            diff = min([np.sum(cv2.absdiff(z, bg[19:29, 19:29])) / 100 for bg in bg_templates])
            if diff > D_GATE: votes += 1
        vector.append(1 if votes >= 2 else 0)
    return vector

def run_v18_surgical_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v18.0 Surgical Persistence Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 4:
        frames_since_last_call += 1
        curr_floor_idx = len(floor_library)
        
        # 1. TIERED LIMITS (v16.0 Baseline)
        refractory_limit = 8 if curr_floor_idx < 40 else 15 if curr_floor_idx < 80 else 25

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: 
            i += 1
            continue

        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)
        despawns = sum(1 for j in range(24) if vec_n[j] == 1 and vec_n1[j] == 0)
        
        is_new_floor = False

        if frames_since_last_call > refractory_limit:
            # TRIGGER A: RESHUFFLE (Catching sparse early floors)
            # Signature: Standard (2+ spawns) OR Sparse (1 spawn + 3 despawns)
            if spawns >= 2 or (spawns == 1 and despawns >= 3):
                # Enforce 2-frame stability for physical layouts
                img_v = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+2]), 0)
                if img_v is not None and get_surgical_vector(img_v, bg_t) == vec_n1:
                    is_new_floor = True

            # TRIGGER B: BOSS-TO-BOSS (99->100 Lowered threshold)
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23:
                grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()
                if grid_delta > 7.0: # Sensitized to catch 99->100
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} | +{spawns}/-{despawns}")
            frames_since_last_call = 0
            i += 1 
            continue

        i += 1

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v18_surgical_audit()