import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
import glob

# --- CONFIGURATION ---
UNIFIED_ROOT = cfg.UNIFIED_INPUTS  # Root directory for unified consensus inputs

def run_census_shifter_sync():
    print("--- ARCHAEOLOGY CENSUS SHIFTER & SYNC (v1.1) ---")
    
    # 1. Target Identification
    run_id = input("Which Run is shifted? (e.g., 3): ").strip()
    target_run_path = os.path.join(UNIFIED_ROOT, f"Run_{run_id}")
    json_path = os.path.join(target_run_path, "final_sequence.json")
    
    if not os.path.exists(target_run_path) or not os.path.exists(json_path):
        print(f"Error: Missing path or JSON at {target_run_path}")
        return

    bad_floor = int(input("Which floor has the double/bad call? (e.g., 61): ").strip())

    # 2. Load Sequence Data
    with open(json_path, 'r') as f:
        sequence = json.load(f)

    # 3. File & Entry Discovery
    all_files = sorted(glob.glob(os.path.join(target_run_path, "F*_frame_*.png")), 
                       key=lambda x: int(os.path.basename(x).split('_')[0][1:]))

    # Find specific files for review
    current_f_img = [f for f in all_files if f.startswith(os.path.join(target_run_path, f"F{bad_floor}_"))]
    next_f_img = [f for f in all_files if f.startswith(os.path.join(target_run_path, f"F{bad_floor + 1}_"))]

    if not current_f_img or not next_f_img:
        print("Could not find the floor sequence images. Check floor numbers.")
        return

    # 4. Visual Verification
    print(f"\nOpening Floor {bad_floor} and {bad_floor + 1} for review...")
    img1 = cv2.imread(current_f_img[0])
    img2 = cv2.imread(next_f_img[0])
    
    # Side-by-side display
    h, w = 500, 400
    comparison = np.hstack((cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))))
    cv2.imshow(f"LEFT: F{bad_floor} | RIGHT: F{bad_floor+1}", comparison)
    cv2.waitKey(1) 

    print("\n[VETO CHECK]")
    print(f" (1) Delete Floor {bad_floor} and shift everything else UP")
    print(f" (2) Delete Floor {bad_floor + 1} and shift everything else UP")
    print(f" (3) Abort")
    
    choice = input("Select action: ").strip()
    cv2.destroyAllWindows()

    if choice not in ['1', '2']:
        print("Operation Aborted.")
        return

    # 5. Execute Shift
    target_del_floor = bad_floor if choice == '1' else bad_floor + 1
    file_to_delete = current_f_img[0] if choice == '1' else next_f_img[0]
    
    # A. Delete Physical File
    print(f"Deleting: {os.path.basename(file_to_delete)}")
    os.remove(file_to_delete)

    # B. Update JSON and Rename Remaining Files
    new_sequence = []
    
    # We iterate through the existing JSON sequence
    for entry in sequence:
        f_num = entry['floor']
        
        if f_num == target_del_floor:
            # Skip the deleted entry
            continue
        
        if f_num > target_del_floor:
            # This entry needs to be shifted up
            old_f_num = f_num
            new_f_num = f_num - 1
            
            # Identify the physical file for this entry
            # Format: F{old_floor}_frame_{original_name}
            old_prefix = f"F{old_f_num}_"
            target_files = [f for f in all_files if os.path.basename(f).startswith(old_prefix)]
            
            if target_files:
                f_path = target_files[0]
                filename = os.path.basename(f_path)
                parts = filename.split("_")
                new_filename = f"F{new_f_num}_" + "_".join(parts[1:])
                new_path = os.path.join(target_run_path, new_filename)
                
                os.rename(f_path, new_path)
                entry['floor'] = new_f_num
                
            if new_f_num % 10 == 0:
                print(f"  Synced: Floor {old_f_num} -> {new_f_num}")

        new_sequence.append(entry)

    # C. Save Updated JSON
    with open(json_path, 'w') as f:
        json.dump(new_sequence, f, indent=4)

    print("\n--- SHIFT & SYNC COMPLETE ---")
    print(f"Updated {json_path} and renamed all subsequent images.")

if __name__ == "__main__":
    run_census_shifter_sync()