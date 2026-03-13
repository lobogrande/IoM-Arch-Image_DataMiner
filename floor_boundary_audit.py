import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v97_Adaptive_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

D_GATE = 7.5         

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v97_adaptive_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    print(f"--- Running v9.7 Adaptive Pulse Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        current_stage_count = len(floor_library)
        
        # 1. DEFINE RULES BY PHASE
        if current_stage_count < 25:
            # PHASE 1: Capture quiet low-density floors
            spawn_min = 1 
            stability_frames = 1
            check_entropy = False
        else:
            # PHASE 2: Block high-floor banner noise
            spawn_min = 3
            stability_frames = 2
            check_entropy = True

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: 
            i += 1
            continue

        vec_n = get_existence_vector(img_n, bg_t)
        vec_n1 = get_existence_vector(img_n1, bg_t)
        
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]

        if len(spawn_slots) >= spawn_min:
            # 2. OPTIONAL ENTROPY (Only Phase 2)
            if check_entropy:
                rows = [divmod(s, 6)[0] for s in spawn_slots]
                if len(set(rows)) == 1 and len(spawn_slots) < 5:
                    i += 1
                    continue

            # 3. STABILITY VALIDATION
            is_stable = True
            for check_off in range(1, stability_frames + 1):
                img_v = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1+check_off]), 0)
                if img_v is None or get_existence_vector(img_v, bg_t) != vec_n1:
                    is_stable = False
                    break
            
            if is_stable:
                floor_num = len(floor_library) + 1
                bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                
                cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_curr, bgr_next)))
                
                floor_library.append({"floor": floor_num, "idx": i, "spawns": len(spawn_slots)})
                print(f" [!] Phase {'1' if current_stage_count < 25 else '2'} | Boundary {floor_num}: Frame {i} -> {i+1}")
                
                i += stability_frames + 1
                continue

        i += 1

    print(f"\n[SUCCESS] Documented {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v97_adaptive_audit()