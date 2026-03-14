import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v11_Consensus_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5         
REFRACTORY_PERIOD = 15 # Hard frames to wait between floor calls

def get_existence_vector(img_gray, bg_templates):
    """Calculates DNA, filtering transient damage number noise."""
    vector = []
    # 1. DAMAGE NUMBER MASKING
    # Use color ranges to mask damage numbers (near-white, yellow-green, red crits)
    # We apply this to the BGR frame, but here we'll use intensity for grayscale simplicity first
    # White crits, yellow crits, red super crits are extreme intensity.
    # For grayscale, anything over 248 is a strong mask candidate.
    _, noise_mask = cv2.threshold(img_gray, 248, 255, cv2.THRESH_BINARY)
    
    # Apply mask: any pixel that is extreme white becomes 'Background' (0) for DNA calculation
    img_gray_clean = cv2.bitwise_and(img_gray, cv2.bitwise_not(noise_mask))

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        # SAMPLING ROI (Slightly reduced to avoid slot edges)
        roi = img_gray_clean[cy-20:cy+20, cx-20:cx+20]
        
        diff = min([np.sum(cv2.absdiff(roi, bg[4:44, 4:44])) / 1600 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_v11_consensus_engine():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_call = 100 # Ensure detection starts immediately
    
    print(f"--- Running v11.0 Systematic Consensus Engine (Run_{TARGET_RUN}) ---")
    start_time = time.time()

    for i in range(len(buffer_files) - 2): # Scan N and N+1
        frames_since_last_call += 1
        
        # Load literal consecutive gray frames
        img_n_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n_gray is None or img_n1_gray is None: continue

        # Voter A/Voter B Baseline (Updated DNA with damage number masking)
        vec_n = get_existence_vector(img_n_gray, bg_t)
        vec_n1 = get_existence_vector(img_n1_gray, bg_t)
        
        diff_count = sum(1 for a, b in zip(vec_n, vec_n1) if a != b)
        sum_n = sum(vec_n)
        sum_n1 = sum(vec_n1)

        # Consensus Decision
        is_new_floor = False
        
        # Gating Voter: Sequential Integrity Guard
        if frames_since_last_call > REFRACTORY_PERIOD:
            
            # SCENARIO 1: Standard Reshuffle (Voter A)
            if diff_count >= 2:
                is_new_floor = True
            
            # SCENARIO 2: Boss-to-Boss Full Floor (Voter B)
            elif sum_n == 24 and sum_n1 == 24:
                # We are Near Floor 98 and both frames are full.
                # Cross-reference the tiny Stage Number Box for co-signature.
                stage_n_roi = img_n_gray[78:105, 115:160]
                stage_n1_roi = img_n1_gray[78:105, 115:160]
                hud_pulse = cv2.absdiff(stage_n_roi, stage_n1_roi).mean()
                
                # Co-signature: Pixel pulse confirmed a stage swap (validated to >2.0 Madrid intensity delta)
                if hud_pulse > 2.0:
                    is_new_floor = True

        if is_new_floor:
            floor_num = len(floor_library) + 1
            
            # Export panels as absolute literal consecutives N and N+1
            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
            
            cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
            cv2.putText(bgr_n1, f"FRAME {i+1} (START)", (30, 50), 0, 0.7, (0,255,0), 2)
            
            handshake = np.hstack((bgr_n, bgr_n1))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}.jpg"), handshake)
            
            floor_library.append({"floor": floor_num, "idx": i})
            print(f" [!] Boundary {floor_num} Logged: Frame {i} -> {i+1} (Consequential consensus)")
            
            frames_since_last_call = 0
            i += 1
            continue

    print(f"\n[SUCCESS] Engine found {len(floor_library)} floors in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v11_consensus_engine()