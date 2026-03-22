# step2_temporal_chunker.py
# Purpose: Execute Master Plan Step 2 - Group frames into distinct floors using 
#          Kinematic rules and Localized DNA Mode.
# Version: 6.5 (The Micro-Block Fix: Consolidating Attack Pauses)

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
    
    print(f"--- STEP 2: KINEMATIC FLOOR GROUPING (v6.5) ---")
    print(f"Processing {len(df)} frames using Game Physics...")

    # 1. SLOT-STRICT MICRO-BLOCKING (v6.5: Increased Gap Tolerance)
    # We increase the gap tolerance to 20 frames (~4 seconds).
    # This prevents minor attack pauses/knockbacks from falsely splitting an attack, 
    # allowing the mode() function to swallow transient fairy noise across the entire engagement.
    df['gap'] = df['frame_idx'].diff().fillna(0)
    df['block_id'] = ((df['slot_id'] != df['slot_id'].shift(1)) | (df['gap'] > 20)).cumsum()
    
    blocks = []
    for block_id, group in df.groupby('block_id'):
        blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'end_frame': int(group['frame_idx'].max()),
            'slot': int(group['slot_id'].iloc[0]),
            'filename': group['filename'].iloc[0],
            # Mode() acts as a localized noise filter within the specific slot attack
            'r4_mode': str(group['r4_dna'].mode()[0]),
            'r3_mode': str(group['r3_dna'].mode()[0]),
            'size': len(group)
        })
    
    print(f"Phase 1: Consolidated frames into {len(blocks)} unified slot-attack blocks.")

    # 2. FLOOR GROUPING (Kinematics & Immutability)
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
        elif b['r4_mode'] != prev_b['r4_mode']:
            is_new_floor = True
            reason = f"R4 DNA Shift ({prev_b['r4_mode']} -> {b['r4_mode']})"
                
        # LAW 3: Row 3 Immutability Broken
        elif b['r3_mode'] != prev_b['r3_mode']:
            is_new_floor = True
            reason = f"R3 DNA Shift ({prev_b['r3_mode']} -> {b['r3_mode']})"

        if is_new_floor:
            b['transition_reason'] = reason
            floors.append(curr_floor)
            curr_floor = [b]
        else:
            curr_floor.append(b)
            
    floors.append(curr_floor)
    print(f"Phase 2: Identified {len(floors)} distinct floors.")

    # 3. EXPORT & VISUAL PROOFS
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