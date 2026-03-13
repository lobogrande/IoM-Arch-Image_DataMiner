import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v95_Tiered_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GLOBAL GATES
D_GATE = 7.5         

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v95_tiered_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    
    print(f"--- Running v9.5 Tiered Multi-Phase Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        current_floor_count = len(floor_library)
        
        # DYNAMIC RULES
        if current_floor_count < 20:
            spawn_min = 2  # High sensitivity for early floors
            stability_needed = 3 # Extra frames of stability to prevent noise
        else:
            spawn_min = 6  # Banner-resistant threshold for high floors
            stability_needed = 2

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: 
            i += 1
            continue

        vec_n = get_existence_vector(img_n, bg_t)
        vec_n1 = get_existence_vector(img_n1, bg_t)
        
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)

        if spawns >= spawn_min:
            # MULTI-FRAME STABILITY VERIFICATION
            is_valid_transition = True
            for check_off in range(1, stability_needed + 1):
                img_v = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1+check_off]), 0)
                if img_v is None or get_existence_vector(img_v, bg_t) != vec_n1:
                    is_valid_transition = False
                    break
            
            if is_valid_transition:
                floor_num = len(floor_library) + 1
                
                # Output literal consecutive handshake
                bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_curr, bgr_next)))
                
                floor_library.append({"floor": floor_num, "idx": i, "spawns": spawns})
                print(f" [!] Phase {'1' if current_floor_count < 20 else '2'} | Boundary {floor_num}: Frame {i} -> {i+1}")
                
                i += stability_needed + 1
                continue

        i += 1

    with open(os.path.join(OUTPUT_DIR, "v95_tiered_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v95_tiered_audit()