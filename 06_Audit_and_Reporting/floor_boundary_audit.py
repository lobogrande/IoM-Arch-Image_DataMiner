import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v26_HomeAnchor_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5         
HUD_PULSE_THRES = 1.1 

def is_player_at_home(img_gray):
    """Detects character pixels in the 'Start of Floor' home position."""
    # ROI: Left of Slot 0 (approx y: 340-380, x: 10-50)
    home_roi = img_gray[340:385, 10:50]
    # The character is dark. Check for a cluster of dark pixels < 50 intensity.
    dark_pixels = np.sum(home_roi < 60)
    return dark_pixels > 150 # Character footprint threshold

def get_lower_surgical_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # Samples BOTTOM quadrant to stay under damage numbers
        roi = img_gray[cy+8:cy+18, cx-5:cx+5]
        diff = min([np.sum(cv2.absdiff(roi, bg[32:42, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v26_home_anchor_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v26.0 Home-Anchor Consensus Engine ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2):
        frames_since_last_call += 1
        curr_floor_count = len(floor_library)
        
        # Tiered Refractory (v16 Baseline)
        ref_limit = 8 if curr_floor_count < 40 else 15 if curr_floor_count < 80 else 25

        img_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n is None or img_n1 is None: continue

        # 1. PRIMARY ANCHOR: Is the player in reset position?
        player_reset = is_player_at_home(img_n1)

        # 2. DATA ANALYSIS
        hud_pulse = cv2.absdiff(img_n[78:105, 115:160], img_n1[78:105, 115:160]).mean()
        vec_n = get_lower_surgical_vector(img_n, bg_t)
        vec_n1 = get_lower_surgical_vector(img_n1, bg_t)
        
        count_n, count_n1 = sum(vec_n), sum(vec_n1)
        spawns = sum(1 for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1)
        despawns = sum(1 for j in range(24) if vec_n[j] == 1 and vec_n1[j] == 0)

        is_new_floor = False
        if frames_since_last_call > ref_limit:
            
            # SIGNATURE 1: The Reset Pop-In (Standard/Quiet)
            # Needs DNA Spawn OR HUD Pulse, but ONLY if Player is at Home
            if player_reset:
                if spawns >= 1 or hud_pulse > HUD_PULSE_THRES:
                    is_new_floor = True

            # SIGNATURE 2: The Step-Down (Boss Clear fix for 99->100)
            # Detects massive ore loss during HUD pulse
            elif count_n >= 22 and count_n1 <= 19 and (hud_pulse > HUD_PULSE_THRES or player_reset):
                is_new_floor = True

            # SIGNATURE 3: The Massive Wipe (Quake Catch-all)
            elif (spawns + despawns) >= 15:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Found (Player Reset: {player_reset})")
            
            frames_since_last_call = 0
            i += 1 
            continue

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v26_home_anchor_audit()