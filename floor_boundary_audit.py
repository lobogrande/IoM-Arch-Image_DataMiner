import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v24_Persistence_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         
HUD_PULSE_THRES = 1.1 

def get_lower_surgical_vector(img_gray, bg_templates):
    """Samples the bottom-center of each slot to stay under damage numbers."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # ROI shifted to the BOTTOM quadrant (cy+8 to cy+18)
        roi = img_gray[cy+8:cy+18, cx-5:cx+5]
        diff = min([np.sum(cv2.absdiff(roi, bg[32:42, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v24_persistence_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v24.0 Step-Down & Persistence Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 3):
        frames_since_last_call += 1
        curr_floor_count = len(floor_library)
        
        # TIERED DYNAMIC LIMITS (v23 Restored)
        ref_limit = 8 if curr_floor_count < 40 else 15 if curr_floor_count < 80 else 25
        spawn_min = 1 if curr_floor_count < 40 else 2

        # 1. CAPTURE DATA (Triple-frame for persistence)
        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        img_n2 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+2]), 0)
        if any(x is None for x in [img_n, img_n1, img_n2]): continue

        # HUD Analysis (Stage counter pulse)
        hud_pulse = cv2.absdiff(img_n[78:105, 115:160], img_n1[78:105, 115:160]).mean()
        
        # DNA Analysis
        vec_n = get_lower_surgical_vector(img_n, bg_t)
        vec_n1 = get_lower_surgical_vector(img_n1, bg_t)
        vec_n2 = get_lower_surgical_vector(img_n2, bg_t)
        
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)
        count_n = sum(vec_n)
        count_n1 = sum(vec_n1)
        
        # STABILITY ANCHOR: Layout must be identical for 2 frames to filter banners/numbers
        is_layout_stable = (vec_n1 == vec_n2)

        is_new_floor = False
        if frames_since_last_call > ref_limit:
            
            # TRIGGER 1: THE STEP-DOWN (Full Floor -> Standard Floor, 99->100 fix)
            if count_n >= 20 and count_n1 <= 18 and hud_pulse > HUD_PULSE_THRES:
                is_new_floor = True
            
            # TRIGGER 2: PERSISTENT SPAWN (Standard Floors)
            elif spawns >= spawn_min and is_layout_stable:
                # Confirm with HUD pulse OR a massive grid shift
                if hud_pulse > HUD_PULSE_THRES or spawns >= 5:
                    is_new_floor = True
            
            # TRIGGER 3: BOSS-TO-BOSS (98->99 Fix)
            elif count_n >= 23 and count_n1 >= 23:
                grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()
                if grid_delta > 8.0:
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} | Pulse: {hud_pulse:.2f}")
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v24_persistence_audit()