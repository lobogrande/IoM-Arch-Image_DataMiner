import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FloorDNA_v72_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 7.5       # Difference from background to count as 'Occupied'
FLOOR_SWAP_MIN = 5 # Minimum simultaneous bit changes to qualify as a NEW FLOOR

def get_occupancy_vector(img_gray, bg_templates):
    """Generates the 24-bit truth of the floor layout."""
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        # Heatmap difference check
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        # BIT: 1 if Ore/Object is present, 0 if Gravel
        vector.append(1 if diff > D_GATE else 0)
    return vector

def run_dna_v72_audit():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    current_floor_start_idx = 0
    active_vector = None
    
    print(f"--- Running v7.2 Logic-Driven Boundary Mapping ---")

    for i in range(len(buffer_files)):
        img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        if img is None: continue
            
        new_vector = get_occupancy_vector(img, bg_t)
        
        if active_vector is None:
            active_vector = new_vector
            continue

        # Calculate bit-wise difference
        # (How many slots changed state since the last frame?)
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)
        
        # LOGIC:
        # If > 5 bits change, it's a Floor Transition (Wipe and Replace)
        # If 1-2 bits change, it's a Mining Event (Stay on same floor, just update vector)
        
        if diff_count >= FLOOR_SWAP_MIN:
            # BOUNDARY DETECTED
            end_idx = i - 1
            floor_num = len(floor_library) + 1
            
            floor_library.append({
                "floor": floor_num,
                "start_idx": current_floor_start_idx,
                "end_idx": end_idx,
                "start_frame": buffer_files[current_floor_start_idx],
                "end_frame": buffer_files[end_idx]
            })

            # Handshake Verification
            p_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[end_idx]))
            c_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
            
            # Label with Stage Numbers for easy scrolling verification
            cv2.putText(p_frame, f"END FLOOR {floor_num}", (40, 60), 0, 0.8, (0,0,255), 2)
            cv2.putText(c_frame, f"START FLOOR {floor_num+1}", (40, 60), 0, 0.8, (0,255,0), 2)
            
            # Mark the changed slots in red to show WHY the transition fired
            for slot in range(24):
                if active_vector[slot] != new_vector[slot]:
                    r, c = divmod(slot, 6)
                    x, y = int(74+(c*59.1))-24, int(261+(r*59.1))-24
                    cv2.rectangle(c_frame, (x, y), (x+48, y+48), (0, 0, 255), 2)

            handshake = np.hstack((p_frame, c_frame))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Boundary_{floor_num}_to_{floor_num+1}.jpg"), handshake)
            
            print(f" [+] Stage {floor_num+1} mapped at frame {i}")
            
            # Reset for next floor
            active_vector = new_vector
            current_floor_start_idx = i
            
        elif diff_count > 0:
            # MINING EVENT: Update the vector so we don't trigger on 'drifting' changes,
            # but do NOT create a new floor.
            active_vector = new_vector

    # Save the Ground Truth Mapping
    with open(os.path.join(OUTPUT_DIR, "v72_boundary_map.json"), "w") as f:
        json.dump(floor_library, f, indent=4)

if __name__ == "__main__":
    run_dna_v72_audit()