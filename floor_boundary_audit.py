import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v93_Spatial_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
RESHUFFLE_MIN = 6    # Minimum 'New Spawns' to consider a reset

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v93_spatial_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    
    print(f"--- Running v9.3 Spatial consensus Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 3:
        if i % 1000 == 0: print(f"  > Frame {i}/{len(buffer_files)}...")

        # 1. State Extraction (Consecutive Triple for Stability)
        imgs = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+3)]
        if any(img is None for img in imgs): 
            i += 1
            continue

        vecs = [get_existence_vector(img, bg_t) for img in imgs]
        
        # 2. Reshuffle Logic
        # A floor reset is: 
        #   A) Multiple NEW ores appear at N+1 (Spawn)
        #   B) The new layout remains STATIC at N+2 (Filters banners)
        spawns = sum(1 for j in range(24) if vecs[0][j] == 0 and vecs[1][j] == 1)
        is_stable = vecs[1] == vecs[2]

        if spawns >= RESHUFFLE_MIN and is_stable:
            floor_num = len(floor_library) + 1
            
            # PANELS: Strict literal consecutives N and N+1
            bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            
            cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            handshake = np.hstack((bgr_curr, bgr_next))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), handshake)
            
            floor_library.append({
                "floor": floor_num,
                "end_idx": i,
                "start_idx": i+1,
                "spawns": spawns
            })
            print(f" [!] Boundary {floor_num}: Frame {i} -> {i+1} | Spawns: {spawns}")
            
            # Advance index past the verified transition
            i += 1 
            continue

        i += 1

    with open(os.path.join(OUTPUT_DIR, "v93_spatial_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v93_spatial_audit()