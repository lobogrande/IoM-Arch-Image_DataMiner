import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v22_HUDAugmented_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         
HUD_PULSE_THRES = 1.0 # Detection threshold for Stage Number redraw

def get_surgical_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # ROI shifted to the LOWER quadrant to avoid fading text/damage numbers
        roi = img_gray[cy+5:cy+15, cx-5:cx+5]
        
        # CROSSHAIR FILTER: If slot 9 is being targeted, use a more tolerant threshold
        local_d_gate = D_GATE + 2.0 if slot == 8 else D_GATE
        
        diff = min([np.sum(cv2.absdiff(roi, bg[29:39, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > local_d_gate else 0)
    return vector

def run_v22_hud_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v22.0 HUD-Augmented Consensus Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2):
        frames_since_last_call += 1
        curr_floor_count = len(floor_library)
        
        # TIERED LIMITS (v19 Framework)
        ref_limit = 10 if curr_floor_count < 40 else 20

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: continue

        # 1. HUD PULSE DETECTION
        # Scanning the Stage counter digits specifically
        hud_roi_n = img_n[78:105, 115:160]
        hud_roi_n1 = img_n1[78:105, 115:160]
        hud_pulse = cv2.absdiff(hud_roi_n, hud_roi_n1).mean()

        # 2. DNA ANALYSIS
        vec_n = get_surgical_vector(img_n, bg_t)
        vec_n1 = get_surgical_vector(img_n1, bg_t)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        despawn_slots = [j for j in range(24) if vec_n[j] == 1 and vec_n1[j] == 0]
        
        is_new_floor = False

        if frames_since_last_call > ref_limit:
            # CROSS-CONFIRMATION LOGIC
            # Signature A: Standard Reshuffle + HUD Confirmation
            if len(spawn_slots) >= 2 and hud_pulse > HUD_PULSE_THRES:
                is_new_floor = True
            
            # Signature B: Massive Wipe (e.g. Floor 11 Quake hit)
            # 10+ ores change state in one frame
            elif (len(spawn_slots) + len(despawn_slots)) >= 10:
                is_new_floor = True
                
            # Signature C: Quiet/Sparse Reset (1-spawn) + HUD Confirmation
            elif len(spawn_slots) >= 1 and hud_pulse > HUD_PULSE_THRES:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Found: Frame {i} -> {i+1} (HUD Pulse: {hud_pulse:.2f})")
            
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v22_hud_audit()