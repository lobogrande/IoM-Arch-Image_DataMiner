# dna_continuity_audit.py
# Purpose: Group frames into stable temporal blocks, identify DNA collisions using 
# Row 4 Ore Identity with spatial masking, and export visual verification frames.

import sys, os, cv2, pandas as pd, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

INPUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_continuity_report.csv")
FINAL_CANDIDATES = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification")

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_row4_ore_identity(img, templates):
    """
    Identifies the 6 ores in Row 4 using spatial masking to focus on 
    the stable center of the grid slot.
    """
    row4_y = int(ORE0_Y + (3 * STEP))
    ore_ids = []
    
    # 20x20 center mask for a 30x30 crop (ignoring noisy outer edges)
    mask_size = 20
    offset = (30 - mask_size) // 2

    for col in range(6):
        x = int(ORE0_X + (col * STEP))
        # Crop the 30x30 slot
        roi = img[row4_y-15 : row4_y+15, x-15 : x+15]
        
        if roi.shape != (30, 30):
            ore_ids.append("err")
            continue

        # Focus on the core 20x20 section
        core = roi[offset : offset+mask_size, offset : offset+mask_size]
        
        best_val = -1
        best_name = "unknown"
        
        for name, tpl in templates.items():
            # Resize template to match masked core if necessary
            if tpl.shape != (mask_size, mask_size):
                tpl_core = tpl[offset : offset+mask_size, offset : offset+mask_size]
            else:
                tpl_core = tpl
                
            res = cv2.matchTemplate(core, tpl_core, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_val:
                best_val = val
                best_name = name
        
        # Using a conservative threshold for the masked core
        ore_ids.append(best_name if best_val > 0.65 else "empty")
        
    return "-".join(ore_ids)

def run_continuity_audit():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run dna_sensor_audit.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    source_dir = cfg.get_buffer_path(0)
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    # Load Ore Templates
    ore_tpls = {}
    ore_dir = os.path.join(cfg.TEMPLATE_DIR, "ores") 
    if os.path.exists(ore_dir):
        for f in os.listdir(ore_dir):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                ore_tpls[name] = cv2.imread(os.path.join(ore_dir, f), 0)

    print(f"--- DNA CONTINUITY & COLLISION RESOLUTION (Masked Mode) ---")

    blocks = []
    current_block = {
        'dna_sig': df.iloc[0]['dna_sig'],
        'start_idx': df.iloc[0]['frame_idx'],
        'end_idx': df.iloc[0]['frame_idx'],
        'filenames': [df.iloc[0]['filename']]
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
                'filenames': [row['filename']]
            }
    blocks.append(current_block)

    report_df = pd.DataFrame(blocks)
    sig_counts = report_df['dna_sig'].value_counts()
    collision_sigs = sig_counts[sig_counts > 1].index.tolist()

    print(f"Resolving {len(collision_sigs)} collision signatures...")

    final_rows = []
    for idx, row in report_df.iterrows():
        identity = "none"
        if row['dna_sig'] in collision_sigs:
            img = cv2.imread(os.path.join(source_dir, row['filenames'][0]), 0)
            if img is not None:
                identity = get_row4_ore_identity(img, ore_tpls)
        
        row_dict = row.to_dict()
        row_dict['full_fingerprint'] = f"{row['dna_sig']}_{identity}"
        final_rows.append(row_dict)

    final_df = pd.DataFrame(final_rows)
    final_df.drop(columns=['filenames']).to_csv(OUT_CSV, index=False)
    
    # Selection of unique floor candidates
    floor_candidates = final_df.sort_values('start_idx').groupby('full_fingerprint').first().reset_index()
    floor_candidates = floor_candidates.sort_values('start_idx')
    
    # VISUAL EXPORT
    print(f"Exporting {len(floor_candidates)} verification frames to {VERIFY_DIR}...")
    for idx, f_row in floor_candidates.iterrows():
        img_path = os.path.join(source_dir, f_row['filenames'][0])
        vis = cv2.imread(img_path)
        if vis is None: continue
        
        # Annotate
        label = f"Start: {f_row['start_idx']} | DNA: {f_row['dna_sig']}"
        ore_str = f"Ores: {f_row['full_fingerprint'].split('_')[1]}"
        cv2.rectangle(vis, (10, 10), (500, 75), (0, 0, 0), -1)
        cv2.putText(vis, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, ore_str, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out_name = f"floor_{idx:03d}_frame_{f_row['start_idx']}.jpg"
        cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)
    
    floor_candidates[['start_idx', 'dna_sig', 'full_fingerprint']].to_csv(FINAL_CANDIDATES, index=False)

    print(f"\n[DONE] Unique Floor Candidates: {len(floor_candidates)}")
    print(f"Visual Proofs saved to: {VERIFY_DIR}")

if __name__ == "__main__":
    run_continuity_audit()