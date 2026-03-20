import json
import os

# --- CONFIG ---
TARGET_RUN = "3"
SHIFT_START_FLOOR = 62  # The first floor that is now 'missing' or needs to be moved up
JSON_PATH = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}/final_sequence.json"

def emergency_json_sync():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        sequence = json.load(f)

    print(f"--- Emergency Sync: Run {TARGET_RUN} ---")
    
    new_sequence = []
    shift_count = 0

    for entry in sequence:
        current_f = entry['floor']
        
        # 1. Detect the duplicate/bad entry that was removed from the folder
        # If the image for F62 was deleted, we remove the JSON entry for 62
        if current_f == SHIFT_START_FLOOR:
            print(f" Removing JSON entry for Floor {current_f} (The deleted duplicate)")
            continue
            
        # 2. Shift everything after the deletion point
        if current_f > SHIFT_START_FLOOR:
            old_f = current_f
            new_f = current_f - 1
            
            # Update the floor number
            entry['floor'] = new_f
            
            # Update the frame filename reference to match the new disk name
            # Original: F63_frame_... -> New: F62_frame_...
            old_frame_name = entry['frame']
            if old_frame_name.startswith(f"F{old_f}_"):
                entry['frame'] = old_frame_name.replace(f"F{old_f}_", f"F{new_f}_", 1)
            
            shift_count += 1
        
        new_sequence.append(entry)

    # Save the repaired JSON
    with open(JSON_PATH, 'w') as f:
        json.dump(new_sequence, f, indent=4)

    print(f"\n[SUCCESS] Shifted {shift_count} entries.")
    print(f"JSON is now synchronized with the physical files in Run_{TARGET_RUN}.")

if __name__ == "__main__":
    emergency_json_sync()