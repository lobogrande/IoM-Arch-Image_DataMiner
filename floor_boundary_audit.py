import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v13_Pulse_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         # Higher threshold to ignore faint particles
REFRACTORY_PERIOD = 15 

def get_existence_vector(img_gray, bg_templates):
    """Calculates DNA using only the 10x10 Micro-Core to avoid damage numbers."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # 10x10 Core only
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        # Compare against core of background
        diff = min([np.sum(cv2.absdiff(roi, bg[19:29, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v13_pulse_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v13.0 Physical Pulse Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2):
        frames_since_last_call += 1
        
        # Load literal consecutives
        img_n_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n_gray is None or img_n1_gray is None: continue

        # 1. PHYSICAL GRID PULSE
        # Check for a massive redraw event across the whole grid area
        grid_roi_n = img_n_gray[230:600, 40:400]
        grid_roi_n1 = img_n1_gray[230:600, 40:400]
        physical_pulse = cv2.absdiff(grid_roi_n, grid_roi_n1).mean()

        # 2. DNA CORE ANALYSIS
        vec_n = get_existence_vector(img_n_gray, bg_t)
        vec_n1 = get_existence_vector(img_n1_gray, bg_t)
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)

        is_new_floor = False

        if frames_since_last_call > REFRACTORY_PERIOD:
            # SIGNATURE A: Standard Reshuffle (Quiet or Loud)
            if spawns >= 2:
                is_new_floor = True
            
            # SIGNATURE B: Boss-to-Boss Pulse (Fix for 98->99)
            # If existence is constant but we see a massive physical redraw spike
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23 and physical_pulse > 8.0:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # Output literal consecutive handshake
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i, "pulse": round(physical_pulse, 2)})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} | Pulse: {physical_pulse:.2f}")
            
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v13_pulse_audit()