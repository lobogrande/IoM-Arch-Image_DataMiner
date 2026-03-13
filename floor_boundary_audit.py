import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v74_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5       
FLOOR_SWAP_MIN = 8 

def load_and_prep_templates(folder, filter_keywords=None, exclude_keywords=None):
    """Safely loads templates while filtering out junk files."""
    prepared = []
    if not os.path.exists(folder): return prepared
    
    for f in os.listdir(folder):
        # Only process image files
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
        # Keyword filtering
        if filter_keywords and not any(k in f for k in filter_keywords): continue
        if exclude_keywords and any(k in f for k in exclude_keywords): continue
        
        img = cv2.imread(os.path.join(folder, f), 0)
        if img is not None:
            prepared.append(cv2.resize(img, (48, 48)))
    return prepared

def get_existence_vector(img_gray, active_templates, shadow_templates, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        
        # 1. Check if it matches Background (Hole/Empty)
        bg_diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        if bg_diff <= D_GATE:
            vector.append(0)
            continue
            
        # 2. Check for Active or Shadow presence
        # If it's NOT gravel, and it matches an active or shadow ore, it exists.
        best_active = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in active_templates] + [0])
        best_shadow = max([cv2.matchTemplate(roi, t, cv2.TM_CCORR_NORMED).max() for t in shadow_templates] + [0])
        
        vector.append(1 if (best_active > 0.82 or best_shadow > 0.82) else 0)
        
    return vector

def run_dna_v74_audit():
    # 1. Robust Asset Loading
    bg_t = load_and_prep_templates("templates", filter_keywords=["background"])
    active_t = load_and_prep_templates("templates", exclude_keywords=["background", "negative", "shadow"])
    shadow_t = load_and_prep_templates("templates", filter_keywords=["shadow"])
    
    print(f"Loaded: {len(bg_t)} BG, {len(active_t)} Active, {len(shadow_t)} Shadow templates.")

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    current_floor_start_idx = 0
    active_existence_vector = None
    
    print(f"--- Running v7.4 Existence-Based Boundary Mapping ---")

    for i in range(len(buffer_files)):
        img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        if img is None: continue
            
        new_vector = get_existence_vector(img, active_t, shadow_t, bg_t)
        
        if active_existence_vector is None:
            active_existence_vector = new_vector
            continue

        # BOUNDARY LOGIC:
        # Measure bits that flipped (Ignoring simple ore-to-shadow transitions)
        diff_count = sum(1 for a, b in zip(active_existence_vector, new_vector) if a != b)
        
        # A floor swap replaces almost everything at once.
        if diff_count >= FLOOR_SWAP_MIN:
            end_idx = i - 1
            floor_num = len(floor_library) + 1
            
            floor_library.append({
                "floor": floor_num,
                "start_idx": current_floor_start_idx,
                "end_idx": end_idx,
                "start_frame": buffer_files[current_floor_start_idx],
                "end_frame": buffer_files[end_idx]
            })

            # Side-by-Side Handshake Visual
            p_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[end_idx]))
            c_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            cv2.putText(p_frame, f"END FLOOR {floor_num}", (40, 60), 0, 0.8, (0,0,255), 2)
            cv2.putText(c_frame, f"START {floor_num+1}", (40, 60), 0, 0.8, (0,255,0), 2)

            handshake = np.hstack((p_frame, c_frame))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num:03}_to_{floor_num+1:03}.jpg"), handshake)
            
            active_existence_vector = new_vector
            current_floor_start_idx = i
            print(f" [+] Boundary Found: Floor {floor_num} -> {floor_num+1}")
            
        elif diff_count > 0:
            # Mining event occurred; update vector but stay on same floor.
            active_existence_vector = new_vector

    with open(os.path.join(OUTPUT_DIR, "v74_existence_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_dna_v74_audit()