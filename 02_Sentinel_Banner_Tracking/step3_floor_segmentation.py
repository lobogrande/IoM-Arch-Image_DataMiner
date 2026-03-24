# step3_floor_segmentation.py
# Purpose: Execute Master Plan Step 3 - Group frames into distinct floors using 
#          Kinematic rules and Strict Row 4 Immutability.
# Version: 3.3 (Dynamic Pathing & Slot-Bound R4 Micro-Despeckle)

import sys, os, cv2, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]

DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"dna_sensor_run_{RUN_ID}.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_start_candidates_run_{RUN_ID}.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_verification_run_{RUN_ID}")

def despeckle_series(series, max_glitch_len=2):
    """
    A surgical micro-filter to remove 1-to-2 frame falling ore glitches on Row 4.
    If the sequence shifts A -> B -> A within 2 frames, B is overwritten as noise.
    Genuine transitions (A -> B -> C) are preserved.
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
        
        # If the duration is very short, check if it reverts
        if 0 < glitch_len <= max_glitch_len:
            if i > 0 and j < n and clean[i-1] == vals[j]:
                # Transient noise detected. Overwrite.
                for k in range(i, j):
                    clean[k] = clean[i-1]
        i = j
    return pd.Series(clean, index=series.index)

def run_temporal_chunking():
    if not os.path.exists(DNA_CSV):
        print(f"Error: {DNA_CSV} not found. Run Step 2 DNA sensor first.")
        return

    # Load DNA data. Force DNA columns to strings to prevent int casting issues.
    df = pd.read_csv(DNA_CSV, dtype={'r3_dna': str, 'r4_dna': str, 'dna_sig': str})
    df = df.sort_values('frame_idx').reset_index(drop=True)
    df['gap'] = df['frame_idx'].diff().fillna(0)
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 3: KINEMATIC FLOOR GROUPING (Run {RUN_ID}) ---")
    print(f"Processing {len(df)} frames using Row 4 Anchoring...")

    # 0. MICRO-DESPECKLE ROW 4 (SLOT-BOUND)
    # We MUST apply this tiny 2-frame filter STRICTLY within a contiguous slot engagement.
    # If applied globally, it falsely erases genuine 1-2 frame floors (like Floor 31) 
    # if the DNA happened to revert after the player teleported.
    df['slot_chunk'] = (df['slot_id'] != df['slot_id'].shift(1)).cumsum()
    
    def clean_r4(g):
        g = g.copy()
        g['r4_clean'] = despeckle_series(g['r4_dna'], max_glitch_len=2)
        return g
        
    df = df.groupby('slot_chunk', group_keys=False).apply(clean_r4)
    df = df.sort_values('frame_idx').reset_index(drop=True)

    # 1. MICRO-BLOCKING (Slot & Cleaned R4)
    # A block breaks ONLY if the slot changes OR the cleaned Row 4 DNA changes.
    df['block_id'] = ((df['slot_id'] != df['slot_id'].shift(1)) | 
                      (df['r4_clean'] != df['r4_clean'].shift(1))).cumsum()
    
    blocks =[]
    for block_id, group in df.groupby('block_id'):
        blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'end_frame': int(group['frame_idx'].max()),
            'slot': int(group['slot_id'].iloc[0]),
            'filename': group['filename'].iloc[0],
            'r4_mode': str(group['r4_clean'].mode()[0]),
            'r3_mode': str(group['r3_dna'].mode()[0]),
            'gap_to_prev': int(group['gap'].iloc[0]),
            'size': len(group)
        })
    
    print(f"Phase 1: Consolidated frames into {len(blocks)} unified slot-attack blocks.")

    # 2. FLOOR GROUPING (Kinematics & R4 Immutability)
    floors = []
    curr_floor = [blocks[0]]
    
    for b in blocks[1:]:
        prev_b = curr_floor[-1]
        
        is_new_floor = False
        reason = ""
        
        # LAW 1: Slot Reversal (Guaranteed Reset)
        # Note: Frame 0 enters with slot -1. Because -1 < any real slot, 
        # this logic will not trigger a false reset on the first actual homing event.
        if b['slot'] < prev_b['slot']:
            is_new_floor = True
            reason = f"Slot Reversal ({prev_b['slot']} -> {b['slot']})"
            
        # LAW 2: Row 4 Immutability Broken
        elif b['r4_mode'] != prev_b['r4_mode']:
            is_new_floor = True
            reason = f"R4 DNA Shift ({prev_b['r4_mode']} -> {b['r4_mode']})"
                
        # LAW 3: The AoE Board Wipe Fallback
        elif b['slot'] == prev_b['slot'] and b['gap_to_prev'] > 60:
            if b['r3_mode'] != prev_b['r3_mode']:
                is_new_floor = True
                reason = f"AoE Board Wipe Detected (R3 DNA Shift after {b['gap_to_prev']}f gap)"

        if is_new_floor:
            b['transition_reason'] = reason
            floors.append(curr_floor)
            curr_floor =[b]
        else:
            curr_floor.append(b)
            
    floors.append(curr_floor)
    print(f"Phase 2: Identified {len(floors)} distinct floors.")

    # 3. EXPORT & VISUAL PROOFS
    print(f"Exporting Step 3 candidates to: {os.path.basename(VERIFY_DIR)}")
    
    # --- CANDIDATE GENERATION ---
    final_candidates =[]
    
    for idx, floor_blocks in enumerate(floors):
        start_block = floor_blocks[0]
        
        candidate = {
            'floor_id': idx + 1,
            'start_frame': start_block['start_frame'],
            'filename': start_block['filename'],
            'slot_id': start_block['slot'],
            'r4_dna_stable': start_block['r4_mode'],
            'r3_dna_stable': start_block['r3_mode'],
            'transition_reason': start_block.get('transition_reason', 'Initial Start (Frame 0)')
        }
        final_candidates.append(candidate)
        
        img_path = os.path.join(SOURCE_DIR, candidate['filename'])
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
    print(f"\n[DONE] Saved {len(final_candidates)} validated start frames to: {os.path.basename(OUT_CSV)}")

if __name__ == "__main__":
    run_temporal_chunking()