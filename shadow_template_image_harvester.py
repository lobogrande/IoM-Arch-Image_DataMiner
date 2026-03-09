import cv2
import numpy as np
import os
import json

# --- 1. THE VERIFIED "TRUTH" COORDINATES ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48 

# --- 2. HARVEST CONFIGURATION ---
# Use this to find those elusive shadow states
HARVEST_TARGETS = [
    {"run": "0", "floor": 67, "offset": 40}, # Look 40 frames AFTER the floor call
    {"run": "0", "floor": 39, "offset": 25},
    {"run": "0", "floor": 103, "offset": 15}
]

UNIFIED_ROOT = "Unified_Consensus_Inputs"

def run_shadow_harvester():
    out_root = "Shadow_Template_Candidates"
    if not os.path.exists(out_root): os.makedirs(out_root)

    for target in HARVEST_TARGETS:
        run_id = target["run"]
        floor_num = target["floor"]
        offset = target["offset"]
        
        run_path = os.path.join(UNIFIED_ROOT, f"Run_{run_id}")
        buffer_path = f"capture_buffer_{run_id}"
        
        # Load sequence to find the starting index
        with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
            sequence = {e['floor']: e for e in json.load(f)}
        
        if floor_num not in sequence: continue
        
        # Calculate the "Shadow Index"
        target_idx = sequence[floor_num]['idx'] + offset
        buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        
        if target_idx >= len(buffer_files): continue
        
        # Load the raw frame from the buffer
        raw_frame = cv2.imread(os.path.join(buffer_path, buffer_files[target_idx]))
        print(f"--- Harvesting Shadows: Floor {floor_num} (Index {target_idx}) ---")

        floor_dir = os.path.join(out_root, f"F{floor_num}_Offset_{offset}")
        if not os.path.exists(floor_dir): os.makedirs(floor_dir)

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx = int(SLOT1_CENTER[0] + (col * STEP_X))
            cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
            
            x1, y1 = cx - 24, cy - 24
            x2, y2 = cx + 24, cy + 24
            
            crop = raw_frame[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(floor_dir, f"S{slot:02}_raw.png"), crop)

if __name__ == "__main__":
    run_shadow_harvester()