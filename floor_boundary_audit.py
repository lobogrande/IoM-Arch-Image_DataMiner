import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
# Output for the visual verification
OUTPUT_DIR = f"diagnostic_results/FloorDNA_Boundaries_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Occupancy Threshold (Delta from background)
D_GATE = 6.0 

def get_floor_dna(img_gray, bg_templates):
    """Generates a 24-bit boolean tuple representing the floor layout."""
    dna = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        # Heatmap-style difference check
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        dna.append(diff > D_GATE)
    return tuple(dna)

def run_dna_boundary_audit():
    # 1. Load Backgrounds
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    
    # 2. Get Sorted Buffer
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    current_floor_start_idx = 0
    current_dna = None
    
    print(f"--- Running v7.0 DNA Boundary Mapping (Run_{TARGET_RUN}) ---")

    for i, fname in enumerate(buffer_files):
        img = cv2.imread(os.path.join(BUFFER_ROOT, fname), 0)
        if img is None: continue
        
        dna = get_floor_dna(img, bg_t)
        
        # Initial floor
        if current_dna is None:
            current_dna = dna
            continue
            
        # Detect Transition (Boundary)
        if dna != current_dna:
            end_idx = i - 1
            
            # Record Floor Data
            floor_entry = {
                "floor_id": len(floor_library),
                "start_idx": current_floor_start_idx,
                "end_idx": end_idx,
                "start_frame": buffer_files[current_floor_start_idx],
                "end_frame": buffer_files[end_idx],
                "dna": current_dna
            }
            floor_library.append(floor_entry)
            
            # --- Generate Visual Handshake ---
            # Panel 1: End of previous floor
            prev_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[end_idx]))
            # Panel 2: Start of new floor
            curr_frame = cv2.imread(os.path.join(BUFFER_ROOT, fname))
            
            # Label them
            cv2.putText(prev_frame, f"END: Floor {len(floor_library)-1}", (50, 50), 0, 0.8, (0,0,255), 2)
            cv2.putText(curr_frame, f"START: Floor {len(floor_library)}", (50, 50), 0, 0.8, (0,255,0), 2)
            
            # Highlight DNA difference in red on the transition frame
            for slot in range(24):
                if dna[slot] != current_dna[slot]:
                    row, col = divmod(slot, 6)
                    x, y = int(74+(col*59.1))-24, int(261+(row*59.1))-24
                    cv2.rectangle(curr_frame, (x,y), (x+48, y+48), (0,0,255), 2)

            handshake = np.hstack((prev_frame, curr_frame))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Transition_{len(floor_library)-1}_to_{len(floor_library)}.jpg"), handshake)
            
            # Reset for next floor
            current_floor_start_idx = i
            current_dna = dna
            
            if len(floor_library) % 10 == 0:
                print(f" [+] Found {len(floor_library)} stage boundaries...")

    # Save final boundary library
    with open(os.path.join(OUTPUT_DIR, "floor_boundary_library.json"), "w") as f:
        json.dump(floor_library, f, indent=4)
        
    print(f"\n[SUCCESS] Documented {len(floor_library)} boundaries. Check {OUTPUT_DIR} for verification images.")

if __name__ == "__main__":
    run_dna_boundary_audit()