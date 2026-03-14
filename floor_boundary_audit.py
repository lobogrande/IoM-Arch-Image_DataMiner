import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v14_Surgical_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.0         
REFRACTORY_PERIOD = 20 # Prunes all redundant calls

def get_surgical_vector(img_gray, bg_templates):
    """Calculates DNA using a triple-ROI consensus per slot to ignore damage numbers."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        
        # Surgical ROIs: Top, Center, Bottom (10x10 each)
        rois = [
            img_gray[cy-15:cy-5, cx-5:cx+5], # Top
            img_gray[cy-5:cy+5, cx-5:cx+5],   # Center
            img_gray[cy+5:cy+15, cx-5:cx+5]  # Bottom
        ]
        
        votes = 0
        for roi in rois:
            # Compare ROI against corresponding part of background templates
            diff = min([np.sum(cv2.absdiff(roi, bg[19:29, 19:29])) / 100 for bg in bg_templates])
            if diff > D_GATE: votes += 1
            
        # Slot consensus: 2/3 regions must show 'Not Gravel'
        vector.append(1 if votes >= 2 else 0)
    return vector

def run_v14_surgical_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v14.0 Surgical Consensus Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2):
        frames_since_last_call += 1
        
        # Load literal consecutives
        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: continue

        # 1. CORE ANALYSIS
        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        is_new_floor = False

        # 2. THE CONSENSUS DECISION
        if frames_since_last_call > REFRACTORY_PERIOD:
            # TRIGGER A: RESHUFFLE (Standard)
            if len(spawn_slots) >= 2:
                # Structural Check: Banners are horizontal lines
                rows = [divmod(s, 6)[0] for s in spawn_slots]
                if len(set(rows)) > 1 or len(spawn_slots) >= 5:
                    is_new_floor = True

            # TRIGGER B: BOSS-TO-BOSS (98->99 Fix)
            # Detect physical grid re-render even if existence counts are identical
            elif sum(vec_n) >= 23 and sum(vec_n1) >= 23:
                grid_delta = cv2.absdiff(img_n[230:600, 40:400], img_n1[230:600, 40:400]).mean()
                if grid_delta > 10.0: # High threshold for physical redraw
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # Output absolute literal consecutive handshake
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Found: Frame {i} -> {i+1}")
            
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v14_surgical_audit()