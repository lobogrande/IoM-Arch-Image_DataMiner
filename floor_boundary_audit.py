import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v20_Anchor_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         

def get_surgical_vector(img_gray, bg_templates):
    """Triple-ROI consensus for physical layout truth."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # 16x16 center core
        roi = img_gray[cy-8:cy+8, cx-8:cx+8]
        diff = min([np.sum(cv2.absdiff(roi, bg[16:32, 16:32])) / 256 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v20_anchor_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v20.0 Temporal Anchor Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 3):
        frames_since_last_call += 1
        curr_floor_idx = len(floor_library)
        
        # 1. TIERED LIMITS (v16/v19 Balance)
        ref_limit = 8 if curr_floor_idx < 40 else 15 if curr_floor_idx < 80 else 25

        # 2. FRAME CAPTURE (Triple capture for stability anchor)
        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        img_n2 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+2]), 0)
        if any(x is None for x in [img_n, img_n1, img_n2]): 
            i += 1
            continue

        grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()

        # SPEED OPTIMIZATION (v19)
        if grid_delta < 1.0: # Lowered from v19 to catch quiet spawns
            continue

        # 3. SURGICAL DNA + STABILITY CHECK
        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        vec_n2 = get_surgical_vector(img_n2, bg_t)
        
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        # The Anchor: The new layout must stay perfectly frozen to be real
        is_layout_stable = (vec_n1 == vec_n2) 

        is_new_floor = False
        if frames_since_last_call > ref_limit:
            # TRIGGER A: PERSISTENT RESHUFFLE
            if len(spawn_slots) >= 1 and is_layout_stable:
                rows = set([divmod(s, 6)[0] for s in spawn_slots])
                
                # Rule T1: Catch quiet early floors (Allows 1 spawn trigger)
                if curr_floor_idx < 40:
                    is_new_floor = True
                # Rule T2/T3: Shield high floors with entropy
                else:
                    if len(spawn_slots) >= 2 and (len(rows) > 1 or len(spawn_slots) >= 5):
                        is_new_floor = True
                    # Exception for the 99->100 End-Game Gap
                    elif curr_floor_idx > 95 and len(spawn_slots) >= 1:
                        is_new_floor = True

            # TRIGGER B: THE FLASH (Catch redraws with 0 DNA delta)
            elif grid_delta > 18.0:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged (Delta: {grid_delta:.2f} | Spawns: {len(spawn_slots)})")
            frames_since_last_call = 0
            i += 1 
            continue

        i += 1

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v20_anchor_audit()