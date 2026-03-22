# step2_temporal_chunker.py
# Purpose: Execute Master Plan Step 2 - Group frames into distinct floors using 
#          Kinematic rules and Slot-Bound DNA Despeckling.
# Version: 6.7 (The Silver Bullet: Slot-Bound Noise Isolation)

import sys, os, cv2, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification")

def despeckle_series(series, max_glitch_len=15):
    """
    Removes transient glitches (A -> B -> A). 
    If a value changes but reverts within max_glitch_len, it is overwritten.
    """
    vals = series.tolist()
    n = len(vals)
    clean = vals.copy()
    i = 0
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        glitch_len = j - i
        
        if 0 < glitch_len <= max_glitch_len:
            if i > 0 and j < n and clean[i-1] == vals[j]:
                for k in range(i, j):
                    clean[k] = clean[i-1]
        i = j
    return clean

def run_temporal_chunking():
    if not os.path.exists(DNA_CSV):
        print(f"Error: {DNA_CSV} not found. Run Step 1 DNA sensor first.")
        return

    # Load DNA data. Force DNA columns to strings to prevent int casting issues.
    df = pd.read_csv(DNA_CSV, dtype={'r3_dna': str, 'r4_dna': str, 'dna_sig': str})
    df = df.sort_values('frame_idx').reset_index(drop=True)
    
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 2: KINEMATIC FLOOR GROUPING (v6.7) ---")
    print(f"Processing {len(df)} frames using Game Physics...")

    # 1. SLOT-BOUND DESPECKLING (The Silver Bullet)
    # We group the data strictly by physical slot engagements. 
    # The despeckle filter is run INSIDE these chunks so it can erase banners/fairies 
    # without EVER crossing a teleport boundary and erasing a fast floor.
    df['slot_chunk'] = (df['slot_id'] != df['slot_id'].shift(1)).cumsum()
    
    def clean_group(g):
        g = g.copy() # Prevent SettingWithCopy warnings
        g['r4_clean'] = despeckle_series(g['r4_dna'], max_glitch_len=15)
        g['r3_clean'] = despeckle_series(g['r3_dna'], max_glitch_len=15)
        return g
        
    df = df.groupby('slot_chunk', group_keys=False).apply(clean_group)
    df = df.sort_values('frame_idx').reset_index(drop=True)
    df['dna_clean'] = df['r4_clean'] + "-" + df['r3_clean']
    
    # 2. MICRO-BLOCKING
    # A block breaks ONLY if the slot changes OR the DESPECKLED DNA changes.
    df['block_id'] = ((df['slot_id'] != df['slot_id'].shift(1)) | 
                      (df['dna_clean'] != df['dna_clean'].shift(1))).cumsum()
    
    blocks = []
    for block_id, group in df.groupby('block_id'):
        blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'end_frame': int(group['frame_idx'].max()),
            'slot': int(group['slot_id'].iloc[0]),
            'filename': group['filename'].iloc[0],
            # Use the clean values for transition logic
            'r4_clean': str(group['r4_clean'].iloc[0]),
            'r3_clean': str(group['r3_clean'].iloc[0]),
            'size': len(group)
        })
    
    print(f"Phase 1: Consolidated frames into {len(blocks)} noise-free attack blocks.")

    # 3. FLOOR GROUPING (Kinematics & Immutability)
    floors = []
    curr_floor = [blocks[0]]
    
    for b in blocks[1:]:
        prev_b = curr_floor[-1]
        
        is_new_floor = False
        reason = ""
        
        # LAW 1: Slot Reversal (Guaranteed Reset)
        if b['slot'] < prev_b['slot']:
            is_new_floor = True
            reason = f"Slot Reversal ({prev_b['slot']} -> {b['slot']})"
            
        # LAW 2: Row 4 Immutability Broken
        elif b['r4_clean'] != prev_b['r4_clean']:
            is_new_floor = True
            reason = f"R4 DNA Shift ({prev_b['r4_clean']} -> {b['r4_clean']})"
                
        # LAW 3: Row 3 Immutability Broken
        elif b['r3_clean'] != prev_b['r3_clean']:
            is_new_floor = True
            reason = f"R3 DNA Shift ({prev_b['r3_clean']} -> {b['r3_clean']})"

        if is_new_floor:
            b['transition_reason'] = reason
            floors.append(curr_floor)
            curr_floor = [b]
        else:
            curr_floor.append(b)
            
    floors.append(curr_floor)
    print(f"Phase 2: Identified {len(floors)} distinct floors.")

    # 4. EXPORT & VISUAL PROOFS
    print(f"Exporting Step 2 candidates to: {VERIFY_DIR}")
    final_candidates = []
    
    for idx, floor_blocks in enumerate(floors):
        start_block = floor_blocks[0]
        
        candidate = {
            'floor_id': idx + 1,
            'start_frame': start_block['start_frame'],
            'filename': start_block['filename'],
            'slot_id': start_block['slot'],
            'r4_dna_stable': start_block['r4_clean'],
            'r3_dna_stable': start_block['r3_clean'],
            'transition_reason': start_block.get('transition_reason', 'Initial Start')
        }
        final_candidates.append(candidate)
        
        # OSD AT BOTTOM (Prevents occluding the Stage ROI header)
        img_path = os.path.join(buffer_dir, candidate['filename'])
        vis = cv2.imread(img_path)
        if vis is not None:
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (10, h - 95), (710, h - 10), (0, 0, 0), -1)
            cv2.putText(vis, f"FLOOR {candidate['floor_id']:03d} | Start Frame: {candidate['start_frame']}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"DNA: {candidate['r4_dna_stable']}-{candidate['r3_dna_stable']} | Slot: {candidate['slot_id']}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, f"Reason: {candidate['transition_reason']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            out_name = f"floor_start_{candidate['floor_id']:03d}_frame_{candidate['start_frame']}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)

    pd.DataFrame(final_candidates).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(final_candidates)} validated start frames to: {OUT_CSV}")

if __name__ == "__main__":
    run_temporal_chunking()