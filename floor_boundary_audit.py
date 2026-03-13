import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v92_Persistence_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
BANNER_INTENSITY = 248

def get_existence_vector(img_gray, bg_templates):
    """Calculates 24-bit DNA layout truth."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v92_persistence_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    
    print(f"--- Running v9.2 State-Persistence Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 3:
        if i % 1000 == 0: print(f"  > Processing {i}/{len(buffer_files)} frames...")

        # 1. State Extraction for N, N+1, N+2 (Consecutive Consensus)
        imgs_gray = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+3)]
        if any(img is None for img in imgs_gray): 
            i += 1
            continue

        vecs = [get_existence_vector(img, bg_t) for img in imgs_gray]
        
        # 2. Reshuffle Analysis
        # Count Spawn events (bits flipping 0 -> 1)
        spawns_1 = sum(1 for j in range(24) if vecs[0][j] == 0 and vecs[1][j] == 1)
        
        # LAYOUT STABILITY CHECK: Real ores stay static for more than 1 frame.
        # Moving banners change the DNA on every single frame.
        is_layout_stable = vecs[1] == vecs[2]

        # 3. HUD SHIELD (Secondary Confirm)
        hud_roi_prev = imgs_gray[0][78:105, 110:170]
        hud_roi_curr = imgs_gray[1][78:105, 110:170]
        
        # Physically block HUD signals if banner pixels are detected in ROI
        is_hud_blocked = np.sum(hud_roi_curr >= BANNER_INTENSITY) > 15
        hud_pulse = cv2.absdiff(hud_roi_curr, hud_roi_prev).mean()

        # 4. TRIGGER DECISION
        is_new_floor = False
        
        # Signature A: Massive DNA Reshuffle + Stable Layout (HUD Independent)
        if spawns_1 >= 10 and is_layout_stable:
            is_new_floor = True
        
        # Signature B: Quiet Reshuffle + Stable Layout + HUD Confirmation (If clear)
        elif spawns_1 >= 3 and is_layout_stable and not is_hud_blocked:
            if hud_pulse > 0.8:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # Export panels as literal consecutives N and N+1
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
                "spawns": spawns_1,
                "hud_pulse": round(hud_pulse, 2)
            })
            print(f" [!] Caught Boundary {floor_num}: Frame {i} -> {i+1} (Stable Consensus)")
            
            i += 1 
            continue

        i += 1

    with open(os.path.join(OUTPUT_DIR, "v92_persistence_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v92_persistence_audit()