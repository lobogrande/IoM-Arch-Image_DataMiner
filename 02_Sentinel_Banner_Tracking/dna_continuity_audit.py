# dna_continuity_audit.py
# Purpose: Group frames into stable temporal blocks, resolve collisions using 
# Circular Masked Block Identity, and export chronologically sorted verification frames.
# Version: 2.9 (Sequential Sorting Update)

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
AI_DIM = 48  # Matching your Unified Tool dimensions

def get_spatial_mask():
    """Matches your original tool: Radius 18 circle in a 48x48 slot."""
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def get_row4_block_identity(img, templates, mask):
    """
    Identifies the 6 blocks in Row 4 using your circular spatial mask.
    """
    row4_y = int(ORE0_Y + (3 * STEP))
    block_ids = []
    
    for col in range(6):
        x = int(ORE0_X + (col * STEP))
        # Crop the 48x48 slot centered on the block
        x1, y1 = int(x - AI_DIM//2), int(row4_y - AI_DIM//2)
        roi = img[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
        
        if roi.shape != (AI_DIM, AI_DIM):
            block_ids.append("err")
            continue

        best_val = -1
        best_name = "unknown"
        
        for name, tpl in templates.items():
            # Ensure template matches dimensions
            if tpl.shape != (AI_DIM, AI_DIM):
                tpl_proc = cv2.resize(tpl, (AI_DIM, AI_DIM))
            else:
                tpl_proc = tpl
                
            # Use masked correlation matching (CCORR) as per your tool
            res = cv2.matchTemplate(roi, tpl_proc, cv2.TM_CCORR_NORMED, mask=mask)
            _, val, _, _ = cv2.minMaxLoc(res)
            
            if val > best_val:
                best_val = val
                best_name = name
        
        block_ids.append(best_name if best_val > 0.80 else "empty")
        
    return "-".join(block_ids)

def run_continuity_audit():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run dna_sensor_audit.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    source_dir = cfg.get_buffer_path(0)
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    mask = get_spatial_mask()
    
    # Load Block Templates
    block_tpls = {}
    block_dir = os.path.join(cfg.TEMPLATE_DIR, "ores") 
    if os.path.exists(ore_dir):
        for f in os.listdir(ore_dir):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                img = cv2.imread(os.path.join(ore_dir, f), 0)
                if img is not None:
                    block_tpls[name] = cv2.resize(img, (AI_DIM, AI_DIM))

    print(f"--- DNA CONTINUITY & COLLISION RESOLUTION (v2.9) ---")

    # 1. Temporal Blocking
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

    # 2. Collision Resolution with Identity
    print(f"Resolving {len(collision_sigs)} collision signatures...")

    final_blocks = []
    for idx, row in report_df.iterrows():
        identity = "none"
        # Always get identity for All-Full cases or collisions
        if row['dna_sig'] in collision_sigs or row['dna_sig'] == "111111-111111":
            img = cv2.imread(os.path.join(source_dir, row['filenames'][0]), 0)
            if img is not None:
                identity = get_row4_block_identity(img, block_tpls, mask)
        
        row_dict = row.to_dict()
        row_dict['full_fingerprint'] = f"{row['dna_sig']}_{identity}"
        final_blocks.append(row_dict)

    # 3. Chronological Candidate Selection
    # Grouping by fingerprint but keeping the EARLIEST block for each
    final_df = pd.DataFrame(final_blocks)
    candidates = final_df.sort_values('start_idx').groupby('full_fingerprint').first().reset_index()
    # CRITICAL: Sort by frame index BEFORE assigning floor sequence number
    candidates = candidates.sort_values('start_idx').reset_index(drop=True)
    
    # 4. Sequential Visual Export
    print(f"Exporting {len(candidates)} verification frames (Sorted)...")
    for idx, f_row in candidates.iterrows():
        img_path = os.path.join(source_dir, f_row['filenames'][0])
        vis = cv2.imread(img_path)
        if vis is None: continue
        
        # OSD Annotation
        # idx + 1 gives us a 1-based floor sequence number
        label = f"FLOOR {idx+1:03d} | Frame: {f_row['start_idx']}"
        sig_label = f"DNA: {f_row['dna_sig']}"
        block_label = f"Identity: {f_row['full_fingerprint'].split('_')[1]}"
        
        cv2.rectangle(vis, (10, 10), (600, 85), (0, 0, 0), -1)
        cv2.putText(vis, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, sig_label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, block_label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        out_name = f"floor_{idx+1:03d}_frame_{f_row['start_idx']}.jpg"
        cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)
    
    candidates[['start_idx', 'dna_sig', 'full_fingerprint']].to_csv(FINAL_CANDIDATES, index=False)
    final_df.drop(columns=['filenames']).to_csv(OUT_CSV, index=False)

    print(f"\n[DONE] Unique Floor Candidates: {len(candidates)}")
    print(f"Chronological Proofs saved to: {VERIFY_DIR}")

if __name__ == "__main__":
    run_continuity_audit()