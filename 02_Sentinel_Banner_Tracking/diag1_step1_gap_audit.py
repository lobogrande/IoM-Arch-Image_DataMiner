# diag_step1_gap_audit.py
# Purpose: Check if missing floors exist in Step 1 results by converting manual filenames to indices.
# Added: Deep Search mode and Surgical Forensic Scan for absolute misses.

import sys, os, cv2, numpy as np, pandas as pd
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
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    return {filename: idx for idx, filename in enumerate(all_files)}, all_files

def perform_surgical_forensic_scan(img_path):
    """Analyzes the raw image for noise (Banners/Fairy) that might block detection."""
    img = cv2.imread(img_path, 0)
    if img is None: return "File Error"
    
    # Analyze Row 3 area (where banners and fairy usually interfere)
    # Target player/ore Y range is roughly 250-450
    roi = img[250:450, :]
    hpp = np.mean(roi, axis=1)
    variance = np.var(roi, axis=1)
    
    avg_intensity = np.mean(hpp)
    max_var = np.max(variance)
    
    # Diagnostics
    # Dark banners drop intensity significantly (< 40)
    # Moving sprites (Fairy) cause high local variance spikes (> 1500)
    if avg_intensity < 40: return f"Blocked: Dark Banner (Intensity: {avg_intensity:.1f})"
    if max_var > 1500: return f"Blocked: High Noise/Fairy (Max Var: {max_var:.1f})"
    return f"Clear but Missed (Int: {avg_intensity:.1f}, Var: {max_var:.1f})"

def run_gap_audit():
    step1_csv = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    discard_csv = "false_negatives_check.csv" 
    
    if not os.path.exists(step1_csv):
        print("Error: Step 1 CSV not found.")
        return

    df_hits = pd.read_csv(step1_csv)
    index_map, all_files = get_buffer_index_map()
    
    df_discards = None
    if os.path.exists(discard_csv):
        df_discards = pd.read_csv(discard_csv)
        df_discards['frame_idx'] = df_discards['filename'].map(index_map)

    if not index_map:
        print("Error: Could not map capture buffer.")
        return

    print("--- STEP 1 GAP AUDIT (Manual Verification) ---")
    print(f"Targeting {len(MANUAL_TARGETS)} manually identified floor starts...")

    for floor, filename in sorted(MANUAL_TARGETS.items()):
        target_idx = index_map.get(filename)
        if target_idx is None: continue

        # 1. Search in Step 1 Successful Hits
        nearby_hits = df_hits[(df_hits['frame_idx'] >= target_idx - 5) & (df_hits['frame_idx'] <= target_idx + 5)]
        if not nearby_hits.empty:
            best_hit = nearby_hits.iloc[0]
            print(f"Floor {floor} (Index {target_idx}): [FOUND] Captured! Conf: {best_hit['confidence']:.4f}")
            continue

        # 2. Search in Discards (Deep Search)
        if df_discards is not None:
            nearby_discards = df_discards[(df_discards['frame_idx'] >= target_idx - 5) & (df_discards['frame_idx'] <= target_idx + 5)]
            if not nearby_discards.empty:
                best_discard = nearby_discards.sort_values('confidence', ascending=False).iloc[0]
                print(f"Floor {floor} (Index {target_idx}): [DEEP SEARCH] Found in discards! Conf: {best_discard['confidence']:.4f}")
                continue

        # 3. Surgical Forensic Scan for Absolute Misses
        img_path = os.path.join(cfg.get_buffer_path(0), filename)
        forensic_result = perform_surgical_forensic_scan(img_path)
        print(f"Floor {floor} (Index {target_idx}): [MISS] {forensic_result}")

if __name__ == "__main__":
    run_gap_audit()