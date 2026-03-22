# step2_temporal_chunker.py
# Purpose: Execute Master Plan Step 2 - Group frames into distinct floors using 
#          Kinematic teleportation rules and 12-Bit DNA Despeckling.
# Version: 6.1 (Pure Data & Kinematic Paradigm - dtype fix)

import sys, os, cv2, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification")

def despeckle_series(series, max_glitch_len=15, passes=2):
    """
    Removes transient glitches (fairies, UI flashes) from a sequence.
    If a value changes but reverts back to the previous value within `max_glitch_len` frames, 
    the glitch is overwritten with the stable surrounding value.
    Genuine transitions (A -> B -> C) are preserved, even if B is only 1 frame long.
    """
    vals = series.tolist()
    for _ in range(passes):
        n = len(vals)
        clean = vals.copy()
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[j] == vals[i]:
                j += 1
            
            glitch_len = j - i
            
            # If the duration is short, check if it reverts to the pre-glitch value
            if 0 < glitch_len <= max_glitch_len:
                if i > 0 and j < n and clean[i-1] == vals[j]:
                    # It's a transient glitch. Overwrite it.
                    for k in range(i, j):
                        clean[k] = clean[i-1]
            i = j
        vals = clean
    return vals

def run_temporal_chunking():
    if not os.path.exists(DNA_CSV):
        print(f"Error: {DNA_CSV} not found. Run Step 1 DNA sensor first.")
        return

    # Load DNA data and ensure chronological sorting.
    # CRITICAL FIX: Force DNA columns to be strings so pandas doesn't parse '001111' as int 1111.
    df = pd.read_csv(DNA_CSV, dtype={'r3_dna': str, 'r4_dna': str, 'dna_sig': str})
    df = df.sort_values('frame_idx').reset_index(drop=True)
    
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 2: KINEMATIC FLOOR GROUPING ---")
    print(f"Processing {len(df)} frames using Game Physics...")

    # 1. DESPECKLE NOISE
    # Clean the Row 3 and Row 4 DNA to silence fairies and scrolling text.
    df['r4_clean'] = despeckle_series(df['r4_dna'], max_glitch_len=15, passes=2)
    df['r3_clean'] = despeckle_series(df['r3_dna'], max_glitch_len=15, passes=2)
    df['dna_clean'] = df['r4_clean'] + "-" + df['r3_clean']
    
    # 2. MICRO-BLOCKING
    # A block breaks ONLY if the slot changes or the cleaned 12-bit DNA changes.
    # No time-gap logic is used, respecting instantaneous teleportation.
    df['block_id'] = ((df['slot_id'] != df['slot_id'].shift(1)) | 
                      (df['dna_clean'] != df['dna_clean'].shift(1))).cumsum()
    
    blocks = []
    for block_id, group in df.groupby('block_id'):
        blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'end_frame': int(group['frame_idx'].max()),
            'slot': int(group['slot_id'].iloc[0]),
            'filename': group['filename'].iloc[0],
            'dna_clean': group['dna_clean'].iloc[0],
            'r4_clean': group['r4_clean'].iloc[0],
            'r3_clean': group['r3_clean'].iloc[0],
            'size': len(group)
        })
    
    print(f"Phase 1: Consolidated {len(df)} frames into {len(blocks)} continuous attack blocks.")

    # 3. FLOOR GROUPING (The Physical Laws)
    floors = []
    curr_floor = [blocks[0]]
    
    for b in blocks[1:]:
        prev_b = curr_floor[-1]
        is_new_floor = False
        reason = ""
        
        # LAW 1: Slot Reversal
        # Step 1 captures only [0, 1, 2, 3, 4, 5, 11]. This is a strictly increasing sequence.
        # Any decrease in this specific subset is a guaranteed reset to a new floor.
        if b['slot'] < prev_b['slot']:
            is_new_floor = True
            reason = f"Slot Reversal (Slot {prev_b['slot']} -> {b['slot']})"
            
        # LAW 2: 12-Bit DNA Shift
        elif b['dna_clean'] != prev_b['dna_clean']:
            is_new_floor = True
            reason = f"12-Bit DNA Shift ({prev_b['dna_clean']} -> {b['dna_clean']})"

        if is_new_floor:
            b['transition_reason'] = reason
            floors.append(curr_floor)
            curr_floor = [b]
        else:
            curr_floor.append(b)
            
    floors.append(curr_floor)
    print(f"Phase 2: Identified {len(floors)} distinct floors using Kinematics and Filtered DNA.")

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
            'dna_clean': start_block['dna_clean'],
            'transition_reason': start_block.get('transition_reason', 'Initial Start')
        }
        final_candidates.append(candidate)
        
        # OSD Generation
        img_path = os.path.join(buffer_dir, candidate['filename'])
        vis = cv2.imread(img_path)
        if vis is not None:
            cv2.rectangle(vis, (10, 10), (650, 85), (0, 0, 0), -1)
            cv2.putText(vis, f"FLOOR {candidate['floor_id']:03d} | Start Frame: {candidate['start_frame']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"12-Bit DNA: {candidate['dna_clean']}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, f"Split Reason: {candidate['transition_reason']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            out_name = f"floor_start_{candidate['floor_id']:03d}_frame_{candidate['start_frame']}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)

    pd.DataFrame(final_candidates).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Master Plan Step 2 Complete.")
    print(f"Saved {len(final_candidates)} start frames to: {OUT_CSV}")
    print("Note: If a player moves forward (e.g., 2 -> 3) AND the 12-Bit DNA perfectly collides,")
    print("      this logic merges them. Further scans may be needed for 1-in-4096 collisions.")

if __name__ == "__main__":
    run_temporal_chunking()