import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import os
import shutil
import glob

# --- CONFIGURATION ---
INPUT_ROOT = "Standardized_Templates_Raw"
OUTPUT_DIR = cfg.TEMPLATE_DIR

# --- MAPPINGS ---
ORE_TYPES = { '1': 'dirt', '2': 'com', '3': 'rare', '4': 'epic', '5': 'leg', '6': 'myth', '7': 'div' }
STATUS_MAP = { '1': 'act', '2': 'sha' }
MOD_MAP    = { '1': 'plain', '2': 'pmod', '3': 'hbar', '4': 'dig', '5': 'xhair', '6': 'fairy', '7': 'noisy' }

def get_next_index(prefix):
    """Checks the templates folder for the next available ID."""
    existing = glob.glob(os.path.join(OUTPUT_DIR, f"{prefix}*.png"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            # Extract the integer between the last underscore and .png
            parts = os.path.basename(f).replace(".png", "").split("_")
            indices.append(int(parts[-1]))
        except: continue
    return max(indices) + 1 if indices else 0

def run_template_architect():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    if not subdirs:
        print("No subdirectories found in Standardized_Templates_Raw.")
        return

    cv2.namedWindow("Architect Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Architect Review", 300, 300)

    for subdir in subdirs:
        subdir_path = os.path.join(INPUT_ROOT, subdir)
        files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            shutil.rmtree(subdir_path)
            continue

        print(f"\n>>> Processing Folder: {subdir}")

        for filename in files:
            filepath = os.path.join(subdir_path, filename)
            img = cv2.imread(filepath)
            if img is None: continue

            # Step 1: Display and Keep/Delete
            cv2.imshow("Architect Review", img)
            print(f"\nFile: {filename}")
            print(" [1] Keep  [2] Delete")
            
            choice = ""
            while choice not in ['1', '2']:
                choice = chr(cv2.waitKey(0) & 0xFF)

            if choice == '2':
                os.remove(filepath)
                print(" Deleted.")
                continue

            # Step 2: Block Type
            print(" Type: [1]dirt [2]com [3]rare [4]epic [5]leg [6]myth [7]div")
            otype_key = ""
            while otype_key not in ORE_TYPES:
                otype_key = chr(cv2.waitKey(0) & 0xFF)
            otype = ORE_TYPES[otype_key]

            # Step 3: Block Tier
            print(" Tier: [1] [2] [3] [4]")
            otier = ""
            while otier not in ['1', '2', '3', '4']:
                otier = chr(cv2.waitKey(0) & 0xFF)

            # Step 4: Status
            print(" Status: [1] act  [2] sha")
            stat_key = ""
            while stat_key not in STATUS_MAP:
                stat_key = chr(cv2.waitKey(0) & 0xFF)
            status = STATUS_MAP[stat_key]

            # Step 5: Modifiers
            mods = []
            print(" Mods: [1]plain [2]pmod [3]hbar [4]dig [5]xhair [6]fairy [7]noisy | [8] DONE")
            while True:
                m_key = chr(cv2.waitKey(0) & 0xFF)
                if m_key == '8': break
                if m_key in MOD_MAP:
                    mods.append(MOD_MAP[m_key])
                    print(f"  + {MOD_MAP[m_key]}")

            # Step 6: Construct Name
            prefix = f"{otype}{otier}_{status}_"
            if mods:
                prefix += "_".join(mods) + "_"
            
            idx = get_next_index(prefix)
            new_name = f"{prefix}{idx}.png"
            
            # Step 7: Move and Rename
            shutil.move(filepath, os.path.join(OUTPUT_DIR, new_name))
            print(f" Moved to: {new_name}")

        # Clean up empty directory
        if not os.listdir(subdir_path):
            os.rmdir(subdir_path)
            print(f"--- Folder {subdir} cleared and removed. ---")

    cv2.destroyAllWindows()
    print("\n[SUCCESS] No more files remain to be renamed. Architect exiting.")

if __name__ == "__main__":
    run_template_architect()