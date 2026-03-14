import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v11_Entropy_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
BANNER_INTENSITY = 248
STABILITY_FRAMES = 2 

def get_existence_vector(img_gray, bg_templates):
    """Calculates 24-bit DNA with an integrated banner-blind mask."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        
        # BANNER SHIELD: If slot is contaminated by white text, mark it as 'Noisy'
        if np.sum(roi >= BANNER_INTENSITY) > 40:
            vector.append(-1) 
            continue
            
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v11_entropy_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_floor = 100
    
    print(f"--- Running v11.0 Entropy-Locked Consensus Engine ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        frames_since_last_floor += 1
        
        # 1. Triple-Frame State Capture
        imgs = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+3)]
        if any(img is None for img in imgs): 
            i += 1
            continue

        vec_n = get_existence_vector(imgs[0], bg_t)
        vec_n1 = get_existence_vector(imgs[1], bg_t)
        vec_n2 = get_existence_vector(imgs[2], bg_t)

        # 2. Transition Signatures
        # Identify slots that just spawned (0 -> 1)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        # 3. CONSENSUS VOTING
        is_new_floor = False
        
        # Voter A: Reshuffle Sensitivity (Catch stages 1-10)
        spawn_min = 1 if len(floor_library) < 10 else 3
        
        if len(spawn_slots) >= spawn_min and frames_since_last_floor > 5:
            # Voter B: Entropy Filter (Reject horizontal lines)
            rows = [divmod(s, 6)[0] for s in spawn_slots]
            is_linear_noise = len(set(rows)) == 1 and len(spawn_slots) < 6
            
            # Voter C: Stability Guard (N1 must stay static at N2)
            if not is_linear_noise and vec_n1 == vec_n2:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # OUTPUT literal consecutive handshake
            bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_curr, bgr_next)))
            
            floor_library.append({"floor": floor_num, "idx": i, "spawns": len(spawn_slots)})
            print(f" [!] Boundary {floor_num} Confirmed: Frame {i} -> {i+1}")
            
            frames_since_last_floor = 0
            i += 2 
            continue

        i += 1

    print(f"\n[SUCCESS] Engine found {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v11_entropy_audit()