import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v10_Consensus_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONSENSUS GATES
D_GATE = 7.5         
MIN_SPAWNS = 3       # Minimum new ores for a valid floor
REFRACTORY_PERIOD = 15 # Hard frames to wait between floor calls

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v10_consensus_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_floor = 100 
    
    print(f"--- Running v10.0 Systematic Consensus Engine ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 4:
        frames_since_last_floor += 1
        
        # 1. Mandatory Sequence Lock
        if frames_since_last_floor < REFRACTORY_PERIOD:
            i += 1
            continue

        # 2. State Extraction (Consecutive Consensus)
        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: 
            i += 1
            continue

        vec_n = get_existence_vector(img_n, bg_t)
        vec_n1 = get_existence_vector(img_n1, bg_t)
        
        # Voter A: Identify New Spawns (0 -> 1)
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)
        # Voter A: Identify Despawns (1 -> 0)
        despawns = sum(1 for j in range(24) if vec_n[j] == 1 and vec_n1[j] == 0)

        # 3. TRANSITION CRITERIA
        # A floor change requires a RE-SHUFFLE (Both spawns and despawns)
        if spawns >= MIN_SPAWNS and despawns >= 2:
            # Voter B: Spatial Stability Check (N+1 must match N+2)
            img_n2 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+2]), 0)
            if img_n2 is not None:
                vec_n2 = get_existence_vector(img_n2, bg_t)
                
                if vec_n1 == vec_n2:
                    floor_num = len(floor_library) + 1
                    
                    # LOGGING
                    bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                    cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_curr, bgr_next)))
                    
                    floor_library.append({"floor": floor_num, "idx": i, "spawns": spawns})
                    print(f" [!] Boundary {floor_num} Confirmed: Frame {i} -> {i+1} | Shuffle: +{spawns}/-{despawns}")
                    
                    frames_since_last_floor = 0
                    i += 2
                    continue

        i += 1

    print(f"\n[SUCCESS] Engine Mapped {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v10_consensus_audit()