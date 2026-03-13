import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v83_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         # Slightly more sensitive to small ores
FLOOR_SWAP_MIN = 8   # Lowered to catch 'similar' floor generation
BANNER_INTENSITY = 248

def is_banner_present(img_gray):
    """Detects scrolling banners in the grid corridor."""
    corridor = img_gray[200:500, 50:400]
    return np.sum(corridor >= BANNER_INTENSITY) > 35 # Slightly more sensitive

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v83_sensitive_audit():
    bg_t = []
    for f in os.listdir("templates"):
        if f.startswith("background"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None: bg_t.append(cv2.resize(img, (48, 48)))

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    active_vector = None
    last_processed_idx = 0
    
    print(f"--- Running v8.3 Sensitive Buffered Audit ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files):
        if i % 500 == 0:
            print(f"  > Progress: {i}/{len(buffer_files)} frames... ({int(i/(time.time()-start_time + 0.1))} FPS)")

        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if curr_bgr is None: 
            i += 1
            continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # 1. SKIP BANNERED FRAMES
        if is_banner_present(curr_gray):
            i += 1
            continue

        # 2. ANALYSIS
        new_vector = get_existence_vector(curr_gray, bg_t)
        
        if active_vector is None:
            active_vector = new_vector
            last_processed_idx = i
            i += 1
            continue

        # Compare bits
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)

        # 3. TRANSITION LOGIC
        if diff_count >= FLOOR_SWAP_MIN:
            # Look for stability: Next 2 banner-free frames must match the NEW layout
            stable_count = 0
            v_idx = i + 1
            
            while stable_count < 2 and v_idx < len(buffer_files):
                v_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[v_idx]), 0)
                if v_img is not None and not is_banner_present(v_img):
                    v_vector = get_existence_vector(v_img, bg_t)
                    if v_vector == new_vector:
                        stable_count += 1
                    else:
                        break # DNA jittered
                v_idx += 1

            if stable_count >= 2:
                floor_num = len(floor_library) + 1
                
                prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[last_processed_idx]))
                cv2.putText(prev_frame, f"END FLOOR {floor_num}", (40, 70), 0, 1.0, (0,0,255), 2)
                cv2.putText(curr_bgr, f"START {floor_num+1}", (40, 70), 0, 1.0, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((prev_frame, curr_bgr)))
                
                active_vector = new_vector
                last_processed_idx = i
                floor_library.append({"floor": floor_num, "idx": i, "frame": buffer_files[i]})
                print(f" [!] Caught Floor {floor_num} -> {floor_num+1} (Frame {i})")
                
                i = v_idx 
                continue

        elif diff_count > 0:
            # Mining event - update the baseline
            active_vector = new_vector
            last_processed_idx = i

        i += 1

    print(f"\n[SUCCESS] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v83_sensitive_audit()