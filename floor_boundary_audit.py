import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v84_FramePerfect_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
FLOOR_SWAP_MIN = 7   # Lowered to catch even minor layout resets
BANNER_INTENSITY = 248

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v84_perfect_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    active_vector = None
    
    print(f"--- Running v8.4 Frame-Perfect Audit (Run_{TARGET_RUN}) ---")
    start_time = time.time()

    for i in range(len(buffer_files)):
        if i % 1000 == 0:
            print(f"  > Processing Frame {i}/{len(buffer_files)}...")

        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if curr_bgr is None: continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        new_vector = get_existence_vector(curr_gray, bg_t)
        
        if active_vector is None:
            active_vector = new_vector
            continue

        # BOUNDARY DETECTION: Count bits that changed since EXACTLY one frame ago
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)

        # If more than 7 ores changed in a single frame, it's a floor swap
        if diff_count >= FLOOR_SWAP_MIN:
            floor_num = len(floor_library) + 1
            
            # PANELS: [I-1] and [I] are absolute consecutives
            prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
            
            cv2.putText(prev_frame, f"FRAME {i-1} (END)", (30, 40), 0, 0.6, (0,0,255), 2)
            cv2.putText(curr_bgr, f"FRAME {i} (START)", (30, 40), 0, 0.6, (0,255,0), 2)
            
            handshake = np.hstack((prev_frame, curr_bgr))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), handshake)
            
            print(f" [!] Boundary {floor_num}: Frame {i-1} -> {i}")
            
            floor_library.append({"floor": floor_num, "idx": i, "frame": buffer_files[i]})
            
            # RESET baseline to the new floor
            active_vector = new_vector
            continue

        # If it was a mining event (1-2 bits), update vector but stay on floor
        elif diff_count > 0:
            active_vector = new_vector

    with open(os.path.join(OUTPUT_DIR, "v84_perfect_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v84_perfect_audit()