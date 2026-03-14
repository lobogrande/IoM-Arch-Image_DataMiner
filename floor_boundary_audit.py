import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v19_SurgicalOptimized_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         

def get_surgical_vector(img_gray, bg_templates):
    """Triple-ROI consensus for combat noise immunity."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # 10x10 Surgical zones
        zones = [img_gray[cy-15:cy-5, cx-5:cx+5], img_gray[cy-5:cy+5, cx-5:cx+5], img_gray[cy+5:cy+15, cx-5:cx+5]]
        votes = 0
        for z in zones:
            diff = min([np.sum(cv2.absdiff(z, bg[19:29, 19:29])) / 100 for bg in bg_templates])
            if diff > D_GATE: votes += 1
        vector.append(1 if votes >= 2 else 0)
    return vector

def run_v19_optimized_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v19.0 Optimized Surgical Auditor ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2):
        frames_since_last_call += 1
        curr_floor_idx = len(floor_library)
        
        # 1. TIERED LIMITS (Restored from v16)
        refractory_limit = 8 if curr_floor_idx < 40 else 15 if curr_floor_idx < 80 else 25

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: continue

        # SPEED OPTIMIZATION: Check global grid delta first
        grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()
        if grid_delta < 2.0: # Skip frames with near-zero activity
            continue

        # 2. SURGICAL ANALYSIS
        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        is_new_floor = False

        if frames_since_last_call > refractory_limit:
            # Standard Spawn Trigger (Min 2 to prevent 1-bit glints)
            if len(spawn_slots) >= 2:
                # Structural Entropy (Row diversity)
                rows = set([divmod(s, 6)[0] for s in spawn_slots])
                if len(rows) > 1 or len(spawn_slots) >= 5:
                    is_new_floor = True

            # Boss-to-Boss Pulse (Lowered to 7.0 for 99->100)
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23 and grid_delta > 7.0:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Found: Frame {i} -> {i+1} (Tier {1 if curr_floor_idx < 40 else 2 if curr_floor_idx < 80 else 3})")
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Mapped {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v19_optimized_audit()