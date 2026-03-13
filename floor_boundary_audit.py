import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v81_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 8.5        
FLOOR_SWAP_MIN = 10  
BANNER_INTENSITY = 248 # Pure white/yellow banner text

def is_banner_present(img_gray):
    """Detects if high-intensity scrolling text is present in the grid area."""
    # Check the central 'corridor' where banners scroll (y: 200 to 500)
    corridor = img_gray[200:500, 50:400]
    # If more than a tiny sliver is pure white, a banner is active
    return np.sum(corridor >= BANNER_INTENSITY) > 50 

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v81_shielded_audit():
    bg_t = []
    for f in os.listdir("templates"):
        if f.startswith("background"):
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None: bg_t.append(cv2.resize(img, (48, 48)))

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    active_vector = None
    last_frame_gray = None
    
    print(f"--- Running v8.1 Banner-Shielded Audit ---")
    start_time = time.time()

    for i, fname in enumerate(buffer_files):
        if i % 250 == 0:
            print(f"  > Scan Progress: {i}/{len(buffer_files)} frames...")

        curr_gray = cv2.imread(os.path.join(BUFFER_ROOT, fname), 0)
        if curr_gray is None: continue

        # 1. BANNER SHIELD: If a banner is detected, skip DNA updates
        if is_banner_present(curr_gray):
            # We skip this frame entirely to prevent banner text from being read as 'Ores'
            continue

        # 2. Baseline initialization
        if active_vector is None:
            active_vector = get_existence_vector(curr_gray, bg_t)
            last_frame_gray = curr_gray
            continue

        # 3. PERFORMANCE TRIGGER (Stage UI check)
        stage_roi = curr_gray[75:110, 80:150]
        prev_stage_roi = last_frame_gray[75:110, 80:150]
        if cv2.absdiff(stage_roi, prev_stage_roi).mean() < 2.5:
            last_frame_gray = curr_gray
            continue

        # 4. TRANSITION CHECK
        new_vector = get_existence_vector(curr_gray, bg_t)
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)

        if diff_count >= FLOOR_SWAP_MIN:
            floor_num = len(floor_library) + 1
            
            # Save the 'Handshake'
            p_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
            c_frame = cv2.imread(os.path.join(BUFFER_ROOT, fname))
            
            cv2.putText(p_frame, f"END FLOOR {floor_num}", (40, 70), 0, 1.0, (0,0,255), 2)
            cv2.putText(c_frame, f"START {floor_num+1}", (40, 70), 0, 1.0, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((p_frame, c_frame)))
            print(f" [!] Verified Stage {floor_num+1} at Frame {i} (Banner-Free)")
            
            active_vector = new_vector
            floor_library.append({"floor": floor_num, "idx": i, "frame": fname})

        last_frame_gray = curr_gray

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v81_shielded_audit()