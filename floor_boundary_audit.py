import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v17_Persistence_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         

def get_surgical_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-8:cy+8, cx-8:cx+8]
        diff = min([np.sum(cv2.absdiff(roi, bg[16:32, 16:32])) / 256 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v17_persistence_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v17.0 Persistence-Validated Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        frames_since_last_call += 1
        curr_floor_idx = len(floor_library)
        
        # 1. TIERED LIMITS
        if curr_floor_idx < 40:
            refractory_limit = 8
            default_stability = 1
        elif curr_floor_idx < 80:
            refractory_limit = 12
            default_stability = 2
        else:
            refractory_limit = 20
            default_stability = 2

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: 
            i += 1
            continue

        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        is_new_floor = False

        # 2. DECISION LOGIC
        if frames_since_last_call > refractory_limit:
            
            # TRIGGER A: RESHUFFLE (Catching the quiet 1-2 spawn transitions)
            if len(spawn_slots) >= 1:
                # DYNAMIC STABILITY: If spawns are low (<=2), require 3-frame "frozen" layout
                # to prove it's an ore and not a moving banner.
                stability_needed = 3 if len(spawn_slots) <= 2 else default_stability
                
                is_stable = True
                for check_off in range(1, stability_needed + 1):
                    img_v = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1+check_off]), 0)
                    if img_v is None or get_surgical_vector(img_v, bg_t) != vec_n1:
                        is_stable = False
                        break
                
                if is_stable:
                    is_new_floor = True

            # TRIGGER B: BOSS-TO-BOSS (98->99 Fix)
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23:
                grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()
                if grid_delta > 10.0:
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} (Spawns: {len(spawn_slots)})")
            
            frames_since_last_call = 0
            i += 1 
            continue

        i += 1

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v17_persistence_audit()