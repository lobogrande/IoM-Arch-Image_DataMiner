import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v90_Consecutive_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
FLOOR_SWAP_MIN = 5   
UI_PULSE_THRES = 0.7 

def get_existence_vector(img_gray, bg_templates):
    """Calculates 24-bit DNA. No banner skipping here."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v90_consecutive_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    active_vector = None
    last_ui_roi = None
    
    print(f"--- Running v9.0 Consecutive Zero-Skip Audit ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 1): # Scan N to N+1
        if i % 1000 == 0: print(f"  > Processing {i}/{len(buffer_files)} frames...")

        # Load Consecutives
        f_curr = buffer_files[i]
        f_next = buffer_files[i+1]
        
        img_curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, f_curr))
        img_next_bgr = cv2.imread(os.path.join(BUFFER_ROOT, f_next))
        if img_curr_bgr is None or img_next_bgr is None: continue
        
        gray_curr = cv2.cvtColor(img_curr_bgr, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(img_next_bgr, cv2.COLOR_BGR2GRAY)

        # 1. State Extraction
        vec_curr = get_existence_vector(gray_curr, bg_t)
        vec_next = get_existence_vector(gray_next, bg_t)
        ui_curr = gray_curr[75:110, 80:180]
        ui_next = gray_next[75:110, 80:180]

        # 2. Delta Analysis
        dna_diff = sum(1 for a, b in zip(vec_curr, vec_next) if a != b)
        ui_pulse = cv2.absdiff(ui_curr, ui_next).mean()

        # 3. TRIGGER: A new floor is born ONLY at the moment N -> N+1
        # Condition: Massive DNA change OR (Pulse + Moderate DNA change)
        if dna_diff >= 15 or (dna_diff >= FLOOR_SWAP_MIN and ui_pulse > UI_PULSE_THRES):
            floor_num = len(floor_library) + 1
            
            # PANELS: These are now 100% literal Frame I and Frame I+1
            cv2.putText(img_curr_bgr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(img_next_bgr, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            handshake = np.hstack((img_curr_bgr, img_next_bgr))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), handshake)
            
            floor_library.append({
                "floor": floor_num,
                "end_idx": i,
                "start_idx": i+1,
                "end_frame": f_curr,
                "start_frame": f_next,
                "dna_delta": dna_diff,
                "ui_pulse": round(ui_pulse, 2)
            })
            print(f" [!] Caught Boundary {floor_num}: Frame {i} -> {i+1} | DNA: {dna_diff} | Pulse: {ui_pulse:.2f}")

    with open(os.path.join(OUTPUT_DIR, "v90_consecutive_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_v90_consecutive_audit()