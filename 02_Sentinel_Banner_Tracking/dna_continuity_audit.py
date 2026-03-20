# dna_continuity_audit.py
# Purpose: Group frames into stable temporal blocks and identify DNA collisions.

import sys, os, pandas as pd, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

INPUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_continuity_report.csv")

def run_continuity_audit():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run dna_sensor_audit.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"--- DNA CONTINUITY AUDIT ({len(df)} frames) ---")

    blocks = []
    if len(df) == 0: return

    # Track the current state of the "Block"
    current_block = {
        'dna_sig': df.iloc[0]['dna_sig'],
        'start_idx': df.iloc[0]['frame_idx'],
        'end_idx': df.iloc[0]['frame_idx'],
        'frame_count': 0,
        'filenames': [df.iloc[0]['filename']],
        'slots_mined': [df.iloc[0]['slot_id']]
    }

    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # CONTINUITY CHECK: Is the DNA still the same?
        if row['dna_sig'] == current_block['dna_sig']:
            current_block['end_idx'] = row['frame_idx']
            current_block['frame_count'] += 1
            current_block['filenames'].append(row['filename'])
            current_block['slots_mined'].append(row['slot_id'])
        else:
            # TRANSITION: Save the completed block and start a new one
            current_block['unique_slots'] = len(set(current_block['slots_mined']))
            blocks.append(current_block)
            
            current_block = {
                'dna_sig': row['dna_sig'],
                'start_idx': row['frame_idx'],
                'end_idx': row['frame_idx'],
                'frame_count': 1,
                'filenames': [row['filename']],
                'slots_mined': [row['slot_id']]
            }

    # Add the final block
    current_block['unique_slots'] = len(set(current_block['slots_mined']))
    blocks.append(current_block)

    # Convert to DataFrame for analysis
    report_df = pd.DataFrame(blocks)
    
    # COLLISION DETECTION: Find blocks that share the same DNA but are NOT contiguous
    # This is where Floor 98 and 99 will be flagged.
    report_df['is_collision'] = report_df.duplicated(subset=['dna_sig'], keep=False)
    
    # Save the report
    report_df.drop(columns=['filenames', 'slots_mined']).to_csv(OUT_CSV, index=False)
    
    print(f"\n[SUMMARY]")
    print(f"Detected Temporal Blocks: {len(report_df)}")
    print(f"Unique DNA Signatures:   {len(report_df['dna_sig'].unique())}")
    print(f"Collision Blocks:        {len(report_df[report_df['is_collision']])}")
    
    print("\n--- TOP COLLISION SIGNATURES (Potential 98/99 Scenarios) ---")
    collisions = report_df[report_df['is_collision']]
    if not collisions.empty:
        print(collisions['dna_sig'].value_counts().head(5))

    print(f"\n[DONE] Continuity report saved to {OUT_CSV}")

if __name__ == "__main__":
    run_continuity_audit()