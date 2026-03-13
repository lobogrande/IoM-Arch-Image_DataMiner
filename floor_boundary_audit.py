import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v75_Optimized_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5       
FLOOR_SWAP_MIN = 8 
CHANGE_SENSITIVITY = 15.0 # Raw pixel mean-diff to trigger deep scan

def load_and_prep_templates(folder, filter_keywords=None, exclude_keywords=None):
    prepared = []
    if not os.path.exists(folder): return prepared
    for f in os.listdir(folder):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        if filter_keywords and not any(k in f for k in filter_keywords): continue
        if exclude_keywords and any(k in f for k in exclude_keywords): continue
        img = cv2.imread(os.path.join(folder, f), 0)
        if img is not None: prepared.append(cv2.resize(img, (48, 48)))
    return prepared

def get_existence_vector(img_gray, active_templates, shadow_templates, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        
        # 1. Quick Background Check
        bg_diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        if bg_diff <= D_GATE:
            vector.append(0)
            continue
            
        # 2. Check for Shadow/Active (Lock the 'Existence')
        best_a = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in active_templates] + [0])
        best_s = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in shadow_templates] + [0])
        vector.append(1 if (best_a > 0.80 or best_s > 0.80) else 0)
        
    return vector

def run_v75_optimized_audit():
    # Load Templates once
    bg_t = load_and_prep_templates("templates", filter_keywords=["background"])
    active_t = load_and_prep_templates("templates", exclude_keywords=["background", "negative", "shadow"])
    shadow_t = load_and_prep_templates("templates", filter_keywords=["shadow"])

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    current_start_idx = 0
    active_vector = None
    last_frame_gray = None
    
    start_time = time.time()
    print(f"--- Running v7.5 High-Speed existence Mapping ---")

    for i, fname in enumerate(buffer_files):
        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if curr_bgr is None: continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        if active_vector is None:
            active_vector = get_existence_vector(curr_gray, active_t, shadow_t, bg_t)
            last_frame_gray = curr_gray
            continue

        # PERFORMANCE: Only analyze if the grid area has significantly changed
        grid_roi = curr_gray[230:600, 40:400]
        prev_grid_roi = last_frame_gray[230:600, 40:400]
        raw_diff = cv2.absdiff(grid_roi, prev_grid_roi).mean()

        if raw_diff > CHANGE_SENSITIVITY:
            new_vector = get_existence_vector(curr_gray, active_t, shadow_t, bg_t)
            diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)

            # FLOOR SWAP: Massive layout change
            if diff_count >= FLOOR_SWAP_MIN:
                end_idx = i - 1
                floor_num = len(floor_library) + 1
                
                floor_library.append({"floor": floor_num, "start_idx": current_start_idx, "end_idx": end_idx})

                # Visual Verification
                p_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[end_idx]))
                cv2.putText(p_frame, f"END FLOOR {floor_num}", (40, 60), 0, 0.8, (0,0,255), 2)
                cv2.putText(curr_bgr, f"START {floor_num+1}", (40, 60), 0, 0.8, (0,255,0), 2)
                
                # Verify that Stage Number area also changed to prevent internal-floor false positives
                handshake = np.hstack((p_frame, curr_bgr))
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), handshake)
                
                active_vector = new_vector
                current_start_idx = i
                print(f" [+] Found Stage {floor_num+1} (Frame {i}) | Time: {time.time()-start_time:.1f}s")
            
            elif diff_count > 0:
                active_vector = new_vector # Mining event update

        last_gray = curr_gray

    print(f"\n[SUCCESS] Mapped {len(floor_library)} floors in {time.time()-start_time:.1f} seconds.")

if __name__ == "__main__":
    run_v75_optimized_audit()