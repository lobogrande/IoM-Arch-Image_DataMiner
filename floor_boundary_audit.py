import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v88_Pulse_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
FLOOR_SWAP_MIN = 5   # Low enough to catch quiet transitions (confirmed by v8.5 scout)
BANNER_INTENSITY = 248
UI_PULSE_THRES = 0.7 # High sensitivity to HUD redraws

def is_banner_present(img_gray):
    """Detects scrolling banners in the grid corridor."""
    corridor = img_gray[220:480, 50:400] # Tightened to avoid HUD/Skills
    return np.sum(corridor >= BANNER_INTENSITY) > 30 

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v88_pulse_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    active_vector = None
    last_ui_roi = None
    last_clean_idx = 0
    
    print(f"--- Running v8.8 High-Sensitivity Pulse Audit ---")
    start_time = time.time()

    for i, fname in enumerate(buffer_files):
        if i % 1000 == 0: print(f"  > Scan Progress: {i}/{len(buffer_files)}...")

        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if curr_bgr is None: continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # 1. SHIELD: Skip frames with banners
        if is_banner_present(curr_gray):
            continue

        new_vector = get_existence_vector(curr_gray, bg_t)
        # Wide ROI for HUD redraw detection
        curr_ui_roi = curr_gray[75:110, 80:180] 
        
        if active_vector is None:
            active_vector = new_vector
            last_ui_roi = curr_ui_roi
            last_clean_idx = i
            continue

        # 2. DELTA CALCULATION
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)
        ui_pulse = cv2.absdiff(curr_ui_roi, last_ui_roi).mean()

        # 3. TRANSITION LOGIC
        # A floor change requires a DNA shift and a HUD pulse
        if diff_count >= FLOOR_SWAP_MIN and ui_pulse > UI_PULSE_THRES:
            floor_num = len(floor_library) + 1
            
            # Use last clean frame before this transition
            prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[last_clean_idx]))
            cv2.putText(prev_frame, f"END FLOOR {floor_num}", (30, 60), 0, 0.8, (0,0,255), 2)
            cv2.putText(curr_bgr, f"START {floor_num+1}", (30, 60), 0, 0.8, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((prev_frame, curr_bgr)))
            print(f" [!] Boundary {floor_num}: Frame {last_clean_idx} -> {i} | Pulse: {ui_pulse:.2f}")
            
            floor_library.append({"floor": floor_num, "idx": i, "frame": fname})
            
            # Reset anchors
            active_vector = new_vector
            last_ui_roi = curr_ui_roi
            last_clean_idx = i
            continue

        # Update anchors on mining events
        elif diff_count > 0:
            active_vector = new_vector
            last_ui_roi = curr_ui_roi
            last_clean_idx = i

    with open(os.path.join(OUTPUT_DIR, "v88_pulse_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v88_pulse_audit()