# diag_step1_gap_audit.py
# Purpose: Check if missing floors exist in Step 1 results by converting manual filenames to indices.
# Added: Deep Search mode to check negative space (discards) for missing hits.

import sys, os, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# Manual start frames identified for the missing floors
MANUAL_TARGETS = {
    99: "frame_20260306_233017_171163.png",
    98: "frame_20260306_232957_176462.png",
    86: "frame_20260306_232754_674016.png",
    77: "frame_20260306_232626_700437.png",
    74: "frame_20260306_232600_716283.png",
    67: "frame_20260306_232449_464173.png",
    49: "frame_20260306_232154_204840.png",
    44: "frame_20260306_232113_173379.png",
    41: "frame_20260306_232043_706382.png",
    35: "frame_20260306_231950_690681.png",
    34: "frame_20260306_231949_674973.png",
    31: "frame_20260306_231946_441753.png",
    29: "frame_20260306_231939_198966.png",
    25: "frame_20260306_231903_218068.png",
    23: "frame_20260306_231842_697467.png",
    17: "frame_20260306_231808_230565.png"
}

def get_buffer_index_map():
    """Scans the capture buffer to build a filename-to-index mapping."""
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(buffer_dir):
        return {}
    
    # Files are strictly sorted to ensure index matches Step 1 logic
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    return {filename: idx for idx, filename in enumerate(all_files)}

def run_gap_audit():
    step1_csv = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    # Using the false_negatives_check.csv generated from the audit for deep search
    discard_csv = "false_negatives_check.csv" 
    
    if not os.path.exists(step1_csv):
        print("Error: Step 1 CSV not found.")
        return

    df_hits = pd.read_csv(step1_csv)
    index_map = get_buffer_index_map()
    
    # Load discards if available for deep search
    df_discards = None
    if os.path.exists(discard_csv):
        df_discards = pd.read_csv(discard_csv)
        # We map filenames to indices to align with the chronological timeline
        df_discards['frame_idx'] = df_discards['filename'].map(index_map)

    if not index_map:
        print("Error: Could not map capture buffer. Check buffer path.")
        return

    print("--- STEP 1 GAP AUDIT (Manual Verification) ---")
    print(f"Targeting {len(MANUAL_TARGETS)} manually identified floor starts...")

    for floor, filename in sorted(MANUAL_TARGETS.items()):
        target_idx = index_map.get(filename)
        
        if target_idx is None:
            print(f"Floor {floor}: [ERROR] Filename {filename} not found in buffer.")
            continue

        # 1. Search in Step 1 Successful Hits
        nearby_hits = df_hits[(df_hits['frame_idx'] >= target_idx - 5) & (df_hits['frame_idx'] <= target_idx + 5)]
        
        if not nearby_hits.empty:
            best_hit = nearby_hits.iloc[0]
            print(f"Floor {floor} (Index {target_idx}): [FOUND] Captured! Conf: {best_hit['confidence']:.4f} (Method: {best_hit['method']})")
            continue

        # 2. Search in Discards (Deep Search)
        if df_discards is not None:
            nearby_discards = df_discards[(df_discards['frame_idx'] >= target_idx - 5) & (df_discards['frame_idx'] <= target_idx + 5)]
            if not nearby_discards.empty:
                # Get the highest confidence discard within the window
                best_discard = nearby_discards.sort_values('confidence', ascending=False).iloc[0]
                print(f"Floor {floor} (Index {target_idx}): [DEEP SEARCH] Found in discards! Conf: {best_discard['confidence']:.4f}")
                print(f"   Reason: Likely below staircase threshold or coordinate drift.")
                continue

        print(f"Floor {floor} (Index {target_idx}): [MISS] Not found in Hits or Discards.")

if __name__ == "__main__":
    run_gap_audit()