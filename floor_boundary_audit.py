import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v21_SurgicalAnchor_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         

def get_lower_surgical_vector(img_gray, bg_templates):
    """Samples the lower-center of each slot to avoid damage number noise."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # ROI shifted to the LOWER quadrant (cy+5 to cy+15)
        roi = img_gray[cy+5:cy+15, cx-5:cx+5]
        diff = min([np.sum(cv2.absdiff(roi, bg[29:39, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v21_surgical_anchor():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v21.0 Surgical Anchor Engine ---")
    start_time = time.time()

    # Scan with a 4-frame window (N, N+1, N+2, N+3)
    for i in range(len(buffer_files) - 4):
        frames_since_last_call += 1
        curr_floor_count = len(floor_library)
        
        # TIERED LIMITS (v16/v19 Framework)
        ref_limit = 10 if curr_floor_count < 40 else 20

        # Load sequence
        imgs = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+4)]
        if any(x is None for x in imgs): continue

        # 1. TRIPLE-CHECK STABILITY CONSENSUS
        # The 'potential' new floor (N+1) MUST persist in N+2 and N+3
        vec_n = get_lower_surgical_vector(imgs[0], bg_t)
        vec_n1 = get_lower_surgical_vector(imgs[1], bg_t)
        vec_n2 = get_lower_surgical_vector(imgs[2], bg_t)
        vec_n3 = get_lower_surgical_vector(imgs[3], bg_t)
        
        # New layout must be identical across 3 frames to be 'real'
        is_layout_persistent = (vec_n1 == vec_n2 == vec_n3)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        is_new_floor = False
        if frames_since_last_call > ref_limit and is_layout_persistent:
            # TRIGGER A: Standard Spawn
            if len(spawn_slots) >= 2:
                rows = set([divmod(s, 6)[0] for s in spawn_slots])
                # Multi-row check to kill linear banner noise
                if len(rows) > 1 or len(spawn_slots) >= 5 or curr_floor_count < 20:
                    is_new_floor = True
            
            # TRIGGER B: Boss-to-Boss Pulse (Sensitized)
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23:
                grid_delta = cv2.absdiff(imgs[0][230:600, 40:400], imgs[1][230:600, 40:400]).mean()
                if grid_delta > 7.0:
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1}")
            
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v21_surgical_anchor()