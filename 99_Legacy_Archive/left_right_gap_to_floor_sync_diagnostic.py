import json
import os

def run_sync_diagnostic():
    # 1. Load the existing confirmed floors
    with open("Run_0_FloorMap_v12.json", "r") as f:
        existing_floors = json.load(f)
    
    # Get a set of indices we've already used
    existing_indices = {entry['idx'] for entry in existing_floors}
    
    # 2. Scan the Gap_Recovery folder for our new finds
    gap_dir = "diagnostic_results/Gap_Recovery_v13_7"
    gap_files = sorted([f for f in os.listdir(gap_dir) if f.endswith(".png")])
    
    new_candidates = []
    for f in gap_files:
        # Filename format: Gap_SlotX_IdxXXXXX.png
        idx = int(f.split("Idx")[1].split(".")[0])
        if idx not in existing_indices:
            new_candidates.append(idx)
            
    print(f"--- Gap-to-Floor Sync Report ---")
    print(f"Current Confirmed Floors: {len(existing_floors)}")
    print(f"Newly Discovered Gaps: {len(new_candidates)}")
    print(f"First 5 New Indices: {new_candidates[:5]}")
    
    if len(new_candidates) > 0:
        print("\nSUCCESS: These gaps represent floors we previously skipped!")
    else:
        print("\nNOTE: All gaps found were already in our confirmed list.")

if __name__ == "__main__":
    run_sync_diagnostic()