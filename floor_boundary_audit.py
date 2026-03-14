import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v21_Hysteresis_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         

def get_surgical_vector(img_gray, bg_templates):
    """Triple-ROI consensus for combat noise immunity."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # 10x10 Surgical zones (Top, Center, Bottom)
        zones = [img_gray[cy-15:cy-5, cx-5:cx+5], img_gray[cy-5:cy+5, cx-5:cx+5], img_gray[cy+5:cy+15, cx-5:cx+5]]
        votes = sum(1 for z in zones if min([np.sum(cv2.absdiff(z, bg[19:29, 19:29])) / 100 for bg in bg_templates]) > D_GATE)
        vector.append(1 if votes >= 2 else 0)
    return vector

def run_v21_hysteresis_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v21.0 Persistence Hysteresis Engine ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        frames_since_last_call += 1
        curr_floor_idx = len(floor_library)
        
        # 1. TIERED REFRACTORY LOCKOUT (v19/v16 Framework)
        ref_limit = 10 if curr_floor_idx < 40 else 20
        
        # 2. FRAME CAPTURE (Quad-capture for 3-frame stability Consensus)
        imgs = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+4)]
        if any(x is None for x in imgs): 
            i += 1
            continue

        # 3. SURGICAL ANALYSIS
        vec_n = get_surgical_vector(imgs[0], bg_t)
        vec_n1 = get_surgical_vector(imgs[1], bg_t)
        vec_n2 = get_surgical_vector(imgs[2], bg_t)
        vec_n3 = get_surgical_vector(imgs[3], bg_t)
        
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        # PERSISTENCE CONSENSUS: New layout must stay frozen for 3 consecutive frames
        is_layout_real = (vec_n1 == vec_n2 == vec_n3)

        is_new_floor = False
        if frames_since_last_call > ref_limit and is_layout_real:
            # TRIGGER A: Standard/Sparse Generation
            # Lowered to 1-spawn trigger now that Persistence Consensus is active
            if len(spawn_slots) >= 1:
                rows = set([divmod(s, 6)[0] for s in spawn_slots])
                
                # Filter Phase 2/3 with entropy; Phase 1 is high-sensitivity
                if curr_floor_idx < 40:
                    is_new_floor = True
                elif len(rows) > 1 or len(spawn_slots) >= 5:
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} (Persistent Spawns: {len(spawn_slots)})")
            frames_since_last_call = 0
            i += 1 
            continue

        i += 1

    print(f"\n[FINISH] Mapped {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v21_hysteresis_audit()