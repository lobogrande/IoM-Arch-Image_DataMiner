import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v73_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5       
FLOOR_SWAP_MIN = 8 # Increased: A new floor usually wipes 15-20+ slots at once

def get_existence_vector(img_gray, active_templates, shadow_templates, bg_templates):
    """
    Generates a 24-bit vector where 1 means 'Something is here' 
    (Active OR Shadow) and 0 means 'Just Gravel'.
    """
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        
        # 1. Check if it's Background Gravel
        bg_diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        if bg_diff <= D_GATE:
            vector.append(0)
            continue
            
        # 2. Check for Active or Shadow presence
        # If it's NOT gravel, we assume it's part of the floor layout
        # We can further verify with templates if needed:
        is_active = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in active_templates]) > 0.85
        is_shadow = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in shadow_templates]) > 0.85
        
        vector.append(1 if (is_active or is_shadow) else 0)
        
    return vector

def run_dna_v73_audit():
    # Load Templates
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    active_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if not any(x in f for x in ["background", "negative", "shadow"])]
    shadow_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if "shadow" in f]
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    current_floor_start_idx = 0
    active_existence_vector = None
    
    print(f"--- Running v7.3 Existence-Based Boundary Mapping ---")

    for i in range(len(buffer_files)):
        img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        if img is None: continue
            
        new_vector = get_existence_vector(img, active_t, shadow_t, bg_t)
        
        if active_existence_vector is None:
            active_existence_vector = new_vector
            continue

        # BOUNDARY LOGIC:
        # We only care about bits flipping from 0 -> 1 (New objects appearing)
        # Or a MASSIVE change in both directions (Floor wipe)
        diff_count = sum(1 for a, b in zip(active_existence_vector, new_vector) if a != b)
        
        # A floor swap usually populates almost the entire grid at once.
        if diff_count >= FLOOR_SWAP_MIN:
            end_idx = i - 1
            floor_num = len(floor_library) + 1
            
            floor_library.append({
                "floor": floor_num,
                "start_idx": current_floor_start_idx,
                "end_idx": end_idx
            })

            # Handshake Verification
            p_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[end_idx]))
            c_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            cv2.putText(p_frame, f"END FLOOR {floor_num}", (40, 60), 0, 0.8, (0,0,255), 2)
            cv2.putText(c_frame, f"START FLOOR {floor_num+1}", (40, 60), 0, 0.8, (0,255,0), 2)

            handshake = np.hstack((p_frame, c_frame))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num}_to_{floor_num+1}.jpg"), handshake)
            
            active_existence_vector = new_vector
            current_floor_start_idx = i
            print(f" [+] Stage {floor_num+1} identified via Layout Reset.")
            
        elif diff_count > 0:
            # Depletion Event: An ore died or the player moved.
            # We update the vector to reflect current state, but we don't call a new floor.
            active_existence_vector = new_vector

    with open(os.path.join(OUTPUT_DIR, "v73_existence_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_dna_v73_audit()