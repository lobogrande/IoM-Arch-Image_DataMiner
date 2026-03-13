import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v82_Buffered_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.0        
FLOOR_SWAP_MIN = 12  # A real swap changes ~20+ bits; banners usually hit 5-8
BANNER_INTENSITY = 248

def is_banner_present(img_gray):
    """Detects scrolling banners in the critical grid 'airspace'."""
    corridor = img_gray[200:500, 50:400]
    return np.sum(corridor >= BANNER_INTENSITY) > 40

def get_existence_vector(img_gray, bg_templates):
    """Calculates 24-bit DNA for the floor."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v82_buffered_audit():
    bg_t = []
    for f in os.listdir("templates"):
        if f.startswith("background"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None: bg_t.append(cv2.resize(img, (48, 48)))

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    active_vector = None
    last_processed_idx = 0
    
    print(f"--- Running v8.2 Buffered Clean-State Audit ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files):
        if i % 500 == 0:
            print(f"  > Frame {i}/{len(buffer_files)}... ({int(i/(time.time()-start_time))} FPS)")

        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if curr_bgr is None: 
            i += 1
            continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # 1. CLEAN CHECK: Skip if banner is present
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

        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)

        # 3. POTENTIAL BOUNDARY
        if diff_count >= FLOOR_SWAP_MIN:
            # We found a potential swap. Verify it's a STABLE NEW FLOOR.
            # We scan forward to find the next 3 banner-free frames and see if DNA matches.
            stable_count = 0
            verification_idx = i + 1
            
            while stable_count < 3 and verification_idx < len(buffer_files):
                v_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[verification_idx]), 0)
                if v_img is not None and not is_banner_present(v_img):
                    v_vector = get_existence_vector(v_img, bg_t)
                    if v_vector == new_vector:
                        stable_count += 1
                    else:
                        break # DNA jittered, not a stable transition
                verification_idx += 1

            if stable_count >= 3:
                # OFFICIAL BOUNDARY
                floor_num = len(floor_library) + 1
                
                prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[last_processed_idx]))
                
                cv2.putText(prev_frame, f"END FLOOR {floor_num}", (40, 70), 0, 1.0, (0,0,255), 2)
                cv2.putText(curr_bgr, f"START {floor_num+1}", (40, 70), 0, 1.0, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((prev_frame, curr_bgr)))
                print(f" [!] Logged Floor {floor_num} -> {floor_num+1} at frame {i}")
                
                active_vector = new_vector
                last_processed_idx = i
                floor_library.append({"floor": floor_num, "idx": i, "frame": buffer_files[i]})
                i = verification_idx # Skip ahead since we verified this window
                continue

        # If it wasn't a swap but there was a mining event, update active_vector
        elif diff_count > 0:
            active_vector = new_vector
            last_processed_idx = i

        i += 1

    print(f"\n[FINISH] Mapped {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v82_buffered_audit()