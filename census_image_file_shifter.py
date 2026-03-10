import cv2
import os
import shutil
import glob

# --- CONFIGURATION ---
UNIFIED_ROOT = "Unified_Consensus_Inputs"

def run_census_shifter():
    print("--- ARCHAEOLOGY CENSUS SHIFTER (v1.0) ---")
    
    # 1. Target Identification
    run_id = input("Which Run is shifted? (e.g., 3): ").strip()
    target_run_path = os.path.join(UNIFIED_ROOT, f"Run_{run_id}")
    
    if not os.path.exists(target_run_path):
        print(f"Error: Path {target_run_path} not found.")
        return

    bad_floor_str = input("Which floor has the double/bad call? (e.g., 61): ").strip()
    bad_floor = int(bad_floor_str)

    # 2. File Discovery
    # We look for files matching the pattern: F{floor}_frame_*.png
    all_files = sorted(glob.glob(os.path.join(target_run_path, "F*_frame_*.png")), 
                       key=lambda x: int(os.path.basename(x).split('_')[0][1:]))

    # Find the current and next floor images for review
    current_f_img = [f for f in all_files if f.startswith(os.path.join(target_run_path, f"F{bad_floor}_"))]
    next_f_img = [f for f in all_files if f.startswith(os.path.join(target_run_path, f"F{bad_floor + 1}_"))]

    if not current_f_img or not next_f_img:
        print("Could not find the floor sequence images. Check floor numbers.")
        return

    # 3. Visual Verification
    print(f"\nOpening Floor {bad_floor} and {bad_floor + 1} for review...")
    img1 = cv2.imread(current_f_img[0])
    img2 = cv2.imread(next_f_img[0])
    
    # Show side-by-side for comparison
    comparison = np.hstack((cv2.resize(img1, (400, 500)), cv2.resize(img2, (400, 500))))
    cv2.imshow(f"LEFT: F{bad_floor} | RIGHT: F{bad_floor+1}", comparison)
    cv2.waitKey(1) # Refresh window

    print("\n[VETO CHECK]")
    print(f" (1) Delete Floor {bad_floor} and shift everything else UP")
    print(f" (2) Delete Floor {bad_floor + 1} and shift everything else UP")
    print(f" (3) Abort")
    
    choice = input("Select action: ").strip()
    cv2.destroyAllWindows()

    if choice not in ['1', '2']:
        print("Operation Aborted.")
        return

    # 4. The Shift Logic
    # Identify the specific file to delete
    file_to_delete = current_f_img[0] if choice == '1' else next_f_img[0]
    start_shift_floor = bad_floor if choice == '1' else bad_floor + 1
    
    print(f"Deleting: {os.path.basename(file_to_delete)}")
    os.remove(file_to_delete)

    # Get remaining files that need renumbering
    # Renaming must be done in REVERSE order if shifting up, 
    # but here we are filling a gap, so we iterate forward.
    remaining_files = [f for f in all_files if f != file_to_delete and int(os.path.basename(f).split('_')[0][1:]) >= start_shift_floor]
    
    print(f"Renaming {len(remaining_files)} subsequent files...")

    for f_path in remaining_files:
        filename = os.path.basename(f_path)
        parts = filename.split("_")
        old_floor_num = int(parts[0][1:])
        new_floor_num = old_floor_num - 1
        
        # Build new filename: F{new_num}_frame_...
        new_filename = f"F{new_floor_num}_" + "_".join(parts[1:])
        new_path = os.path.join(target_run_path, new_filename)
        
        os.rename(f_path, new_path)
        if new_floor_num % 10 == 0:
            print(f"  Fixed: Floor {old_floor_num} -> Floor {new_floor_num}")

    # 5. Final JSON Adjustment
    # Since the image filenames changed, the JSON 'frame' references might now be wrong
    # We should alert the user that the JSON needs a rebuild or update.
    print("\n--- SHIFT COMPLETE ---")
    print("Renamed all files to the end of the dataset.")
    print("[!] NOTE: You must now re-run the final_sequence.json generator for this Run")
    print("    so that the JSON indices match the new file names.")

if __name__ == "__main__":
    import numpy as np
    run_census_shifter()