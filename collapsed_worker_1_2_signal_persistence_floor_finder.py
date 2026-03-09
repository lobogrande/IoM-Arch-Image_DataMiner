import cv2
import numpy as np
import os
import json

# --- MASTER CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
COLLAPSED_OUTPUT = "Collapsed_Discovery"

def run_collapsed_worker():
    if not os.path.exists(COLLAPSED_OUTPUT): os.makedirs(COLLAPSED_OUTPUT)

    for ds_id in DATASETS:
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(buffer_path): continue
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        ds_out = os.path.join(COLLAPSED_OUTPUT, f"Run_{ds_id}")
        if os.path.exists(ds_out): shutil.rmtree(ds_out)
        os.makedirs(ds_out)

        print(f"--- COLLAPSED SCAN RUN {ds_id} ---")
        
        # Initialization: Assume Floor 1 is Frame 0
        sequence = [{'idx': 0, 'floor': 1, 'frame': frames[0]}]
        
        prev_h = cv2.imread(os.path.join(buffer_path, frames[0]), 0)[52:76, 100:142]
        prev_c = cv2.imread(os.path.join(buffer_path, frames[0]), 0)[230:246, 250:281]
        
        current_floor = 1
        for i in range(1, len(frames) - 5):
            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[52:76, 100:142]; curr_c = img[230:246, 250:281]
            
            # Pulse + Multi-Frame Persistence Logic
            if np.mean(cv2.absdiff(curr_h, prev_h)) > 2.2 and np.mean(cv2.absdiff(curr_c, prev_c)) > 1.5:
                is_permanent = True
                for offset in range(1, 4):
                    future_h = cv2.imread(os.path.join(buffer_path, frames[i+offset]), 0)[52:76, 100:142]
                    if np.mean(cv2.absdiff(future_h, curr_h)) > 5.0:
                        is_permanent = False; break
                
                if is_permanent:
                    current_floor += 1
                    anchor_idx = i + 5
                    sequence.append({'idx': anchor_idx, 'floor': current_floor, 'frame': frames[anchor_idx]})
                    
                    # Log finding
                    if current_floor % 10 == 0: print(f" Discovered F{current_floor} @ index {anchor_idx}")
                    
                    # Update Baselines
                    prev_h = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[52:76, 100:142]
                    prev_c = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[230:246, 250:281]
                    continue
            
            prev_h, prev_c = curr_h, curr_c

        # Save results
        with open(os.path.join(ds_out, "collapsed_sequence.json"), 'w') as f:
            json.dump(sequence, f, indent=4)
            
        # Copy images for review
        for entry in sequence:
            shutil.copy2(os.path.join(buffer_path, entry['frame']), 
                         os.path.join(ds_out, f"F{entry['floor']}_{entry['frame']}"))

if __name__ == "__main__":
    run_collapsed_worker()