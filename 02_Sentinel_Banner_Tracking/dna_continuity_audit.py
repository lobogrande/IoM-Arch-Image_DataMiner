# dna_continuity_audit.py
# Purpose: Group frames into stable temporal blocks and identify DNA collisions using Row 4 Ore Identity.

import sys, os, cv2, pandas as pd, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

INPUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_continuity_report.csv")
FINAL_CANDIDATES = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_row4_ore_identity(img, templates):
    """
    Identifies the 6 ores in Row 4 to create a high-fidelity 'Identity String'.
    Only used to resolve collisions where Background DNA is identical.
    """
    row4_y = int(ORE0_Y + (3 * STEP))
    ore_ids = []
    
    for col in range(6):
        x = int(ORE0_X + (col * STEP))
        # Crop a 30x30 ore sample
        roi = img[row4_y-15 : row4_y+15, x-15 : x+15]
        
        best_val = -1
        best_name = "unknown"
        
        for name, tpl in templates.items():
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_val:
                best_val = val
                best_name = name
        
        ore_ids.append(best_name if best_val > 0.6 else "empty")
        
    return "-".join(ore_ids)

def run_continuity_audit():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run dna_sensor_audit.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    source_dir = cfg.get_buffer_path(0)
    
    # Load Ore Templates for Row 4 Resolution
    ore_tpls = {}
    ore_dir = os.path.join(cfg.TEMPLATE_DIR, "ores") # Assuming an 'ores' subfolder exists
    if os.path.exists(ore_dir):
        for f in os.listdir(ore_dir):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                ore_tpls[name] = cv2.imread(os.path.join(ore_dir, f), 0)

    print(f"--- DNA CONTINUITY & COLLISION RESOLUTION ---")

    blocks = []
    current_block = {
        'dna_sig': df.iloc[0]['dna_sig'],
        'start_idx': df.iloc[0]['frame_idx'],
        'end_idx': df.iloc[0]['frame_idx'],
        'filenames': [df.iloc[0]['filename']],
        'row4_identity': ""
    }

    for i in range(1, len(df)):
        row = df.iloc[i]
        if row['dna_sig'] == current_block['dna_sig']:
            current_block['end_idx'] = row['frame_idx']
            current_block['filenames'].append(row['filename'])
        else:
            blocks.append(current_block)
            current_block = {
                'dna_sig': row['dna_sig'],
                'start_idx': row['frame_idx'],
                'end_idx': row['frame_idx'],
                'filenames': [row['filename']],
                'row4_identity': ""
            }
    blocks.append(current_block)

    # RESOLVE COLLISIONS
    # If a dna_sig appears more than once in the blocks list, we check Row 4 Identity
    report_df = pd.DataFrame(blocks)
    sig_counts = report_df['dna_sig'].value_counts()
    collision_sigs = sig_counts[sig_counts > 1].index.tolist()

    print(f"Resolving {len(collision_sigs)} collision signatures using Row 4 Ore Identity...")

    final_rows = []
    for idx, row in report_df.iterrows():
        identity = "none"
        if row['dna_sig'] in collision_sigs:
            # Sample the very first frame of this block to get the ore identity
            img = cv2.imread(os.path.join(source_dir, row['filenames'][0]), 0)
            if img is not None:
                identity = get_row4_ore_identity(img, ore_tpls)
        
        row_dict = row.to_dict()
        row_dict['full_fingerprint'] = f"{row['dna_sig']}_{identity}"
        final_rows.append(row_dict)

    final_df = pd.DataFrame(final_rows)
    
    # Save the full continuity report
    final_df.drop(columns=['filenames']).to_csv(OUT_CSV, index=False)
    
    # PRODUCE FLOOR START CANDIDATES
    # We take the first frame of every UNIQUE full_fingerprint
    # We sort by start_idx to maintain floor order
    floor_candidates = final_df.sort_values('start_idx').groupby('full_fingerprint').first().reset_index()
    floor_candidates = floor_candidates.sort_values('start_idx')
    
    floor_candidates[['start_idx', 'dna_sig', 'full_fingerprint']].to_csv(FINAL_CANDIDATES, index=False)

    print(f"\n[SUMMARY]")
    print(f"Total Temporal Blocks:    {len(final_df)}")
    print(f"Unique Floor Candidates:  {len(floor_candidates)}")
    print(f"Candidates saved to:      {FINAL_CANDIDATES}")

if __name__ == "__main__":
    run_continuity_audit()