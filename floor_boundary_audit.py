import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v101_Signature_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
MIN_SPAWNS = 2       # High sensitivity to catch quiet 1-10 floors
STABILITY_FRAMES = 2 # Layout must remain static
REFRACTORY_PERIOD = 8 # Lowered to prevent skipping fast floors

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v101_signature_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100
    
    print(f"--- Running v10.1 Structural Signature Auditor ---")
    start_time = time.time()

    i = 0
    while i < len(buffer_files) - 5:
        frames_since_last_call += 1
        
        # 1. Capture Triple for Stability
        imgs = [cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[x]), 0) for x in range(i, i+3)]
        if any(img is None for img in imgs): 
            i += 1
            continue

        vec_n = get_existence_vector(imgs[0], bg_t)
        vec_n1 = get_existence_vector(imgs[1], bg_t)
        vec_n2 = get_existence_vector(imgs[2], bg_t)

        # 2. ANALYSIS
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        # Stability: New layout must stay fixed for Voter B
        is_stable = vec_n1 == vec_n2
        
        # HUD Pulse: Check Stage Number digits for Voter C
        hud_n = imgs[0][78:105, 115:160]
        hud_n1 = imgs[1][78:105, 115:160]
        hud_pulse = cv2.absdiff(hud_n, hud_n1).mean()

        # 3. SIGNATURE CONSENSUS
        is_new_floor = False
        
        if len(spawn_slots) >= MIN_SPAWNS and frames_since_last_call > REFRACTORY_PERIOD:
            # Check Structural Fingerprint (Reject horizontal lines/banners)
            rows = [divmod(s, 6)[0] for s in spawn_slots]
            is_linear = len(set(rows)) == 1 and len(spawn_slots) < 5
            
            if is_stable and not is_linear:
                is_new_floor = True
            
            # Tie-breaker: If stability is broken (character moved), but HUD pulsed significantly
            elif hud_pulse > 2.5:
                is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # Save literal consecutive handshake
            bgr_curr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_next = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            cv2.putText(bgr_curr, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_next, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), np.hstack((bgr_curr, bgr_next)))
            
            floor_library.append({"floor": floor_num, "idx": i, "spawns": len(spawn_slots), "pulse": round(hud_pulse, 2)})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} | Pulse: {hud_pulse:.2f}")
            
            frames_since_last_call = 0
            i += 2
            continue

        i += 1

    print(f"\n[FINISH] Auditor found {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v101_signature_audit()