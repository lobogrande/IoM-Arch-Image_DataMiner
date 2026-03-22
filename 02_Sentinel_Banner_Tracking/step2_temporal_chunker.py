# step2_temporal_chunker.py
# Purpose: Execute Master Plan Step 2 - Group frames into distinct floors using 
#          Kinematic rules and Temporal Stability Gates.
# Version: 6.8 (The Temporal Hammer: Frame Locking & DNA Debouncing)

import sys, os, cv2, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification")

def run_temporal_chunking():
    if not os.path.exists(DNA_CSV):
        print(f"Error: {DNA_CSV} not found. Run Step 1 DNA sensor first.")
        return

    # Load DNA data. Force DNA columns to strings to prevent int casting issues.
    df = pd.read_csv(DNA_CSV, dtype={'r3_dna': str, 'r4_dna': str, 'dna_sig': str})
    df = df.sort_values('frame_idx').reset_index(drop=True)
    
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 2: KINEMATIC FLOOR GROUPING (v6.8) ---")
    print(f"Processing {len(df)} frames using Temporal Physics...")

    # 1. TEMPORAL STABILITY PASS
    # We identify "Dirty" transitions where the DNA is flickering
    df['dna_full'] = df['r4_dna'] + "-" + df['r3_dna']
    df['frame_diff'] = df['frame_idx'].diff().fillna(0)
    
    # 2. MICRO-BLOCKING (v6.8 Stability Update)
    # A block breaks ONLY if:
    # A) The slot changes
    # B) The frame gap is > 1 (Physical teleport possibility) AND the DNA has shifted
    # C) The DNA has been stable for a few frames (Debouncing)
    
    blocks = []
    curr_group = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            curr_group.append(row)
            continue
            
        prev = df.iloc[i-1]
        
        # Physical impossibility check: consecutive frames cannot be different floors
        is_consecutive = (row['frame_idx'] - prev['frame_idx'] == 1)
        slot_changed = (row['slot_id'] != prev['slot_id'])
        dna_changed = (row['dna_full'] != prev['dna_full'])
        
        split_needed = False
        if slot_changed:
            split_needed = True
        elif dna_changed and not is_consecutive:
            # DNA changed and there's a gap (potential floor transition)
            # Check for stability: does this DNA last for at least 3 frames?
            if i + 3 < len(df):
                future_dna = df.iloc[i:i+3]['dna_full'].unique()
                if len(future_dna) == 1: # It is stable
                    split_needed = True
        
        if split_needed:
            blocks.append(pd.DataFrame(curr_group))
            curr_group = [row]
        else:
            curr_group.append(row)
            
    if curr_group:
        blocks.append(pd.DataFrame(curr_group))

    # 3. BLOCK CONSOLIDATION
    consolidated_blocks = []
    for group in blocks:
        consolidated_blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'end_frame': int(group['frame_idx'].max()),
            'slot': int(group['slot_id'].iloc[0]),
            'filename': group['filename'].iloc[0],
            'r4_mode': str(group['r4_dna'].mode()[0]),
            'r3_mode': str(group['r3_dna'].mode()[0]),
            'dna_full': str(group['r4_dna'].mode()[0]) + "-" + str(group['r3_dna'].mode()[0])
        })
    
    print(f"Phase 1: Consolidated frames into {len(consolidated_blocks)} stable blocks.")

    # 4. FLOOR GROUPING (The Physical Laws)
    floors = []
    curr_floor = [consolidated_blocks[0]]
    
    for b in consolidated_blocks[1:]:
        prev_b = curr_floor[-1]
        is_new_floor = False
        reason = ""
        
        # LAW 1: Slot Reversal (Primary Reset Signal)
        if b['slot'] < prev_b['slot']:
            is_new_floor = True
            reason = f"Slot Reversal ({prev_b['slot']} -> {b['slot']})"
            
        # LAW 2: Stable DNA Shift
        elif b['dna_full'] != prev_b['dna_full']:
            is_new_floor = True
            reason = f"DNA Shift ({prev_b['dna_full']} -> {b['dna_full']})"

        if is_new_floor:
            b['transition_reason'] = reason
            floors.append(curr_floor)
            curr_floor = [b]
        else:
            curr_floor.append(b)
            
    floors.append(curr_floor)
    print(f"Phase 2: Identified {len(floors)} distinct floors using Stability Gates.")

    # 5. EXPORT & VISUAL PROOFS
    print(f"Exporting Step 2 candidates to: {VERIFY_DIR}")
    final_candidates = []
    
    for idx, floor_blocks in enumerate(floors):
        start_block = floor_blocks[0]
        
        candidate = {
            'floor_id': idx + 1,
            'start_frame': start_block['start_frame'],
            'filename': start_block['filename'],
            'slot_id': start_block['slot'],
            'r4_dna_stable': start_block['r4_mode'],
            'r3_dna_stable': start_block['r3_mode'],
            'transition_reason': start_block.get('transition_reason', 'Initial Start')
        }
        final_candidates.append(candidate)
        
        img_path = os.path.join(buffer_dir, candidate['filename'])
        vis = cv2.imread(img_path)
        if vis is not None:
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (10, h - 95), (750, h - 10), (0, 0, 0), -1)
            cv2.putText(vis, f"FLOOR {candidate['floor_id']:03d} | Frame: {candidate['start_frame']}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"DNA: {candidate['r4_dna_stable']}-{candidate['r3_dna_stable']} | Reason: {candidate['transition_reason']}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            out_name = f"floor_start_{candidate['floor_id']:03d}_frame_{candidate['start_frame']}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)

    pd.DataFrame(final_candidates).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(final_candidates)} start frames to: {OUT_CSV}")

if __name__ == "__main__":
    run_temporal_chunking()