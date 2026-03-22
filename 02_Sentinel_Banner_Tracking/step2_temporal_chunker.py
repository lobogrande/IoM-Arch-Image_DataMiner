# step2_temporal_chunker.py
# Purpose: Execute Master Plan Step 2 - Group frames into distinct floors using 
#          Kinematic rules and Stage-ROI Visual Verification.
# Version: 6.3 (The Visual Arbiter: ROI Validation & UI Relocation)

import sys, os, cv2, pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification")

# Approximate Stage ROI (Area where "Stage X-Y" is displayed)
# We use this to confirm if the floor number actually changed.
STAGE_ROI_BOX = (15, 10, 180, 60)  # x, y, w, h

def get_stage_roi(img):
    """Extracts the top-left stage header as a grayscale comparison block."""
    if img is None: return None
    x, y, w, h = STAGE_ROI_BOX
    roi = img[y:y+h, x:x+w]
    if roi.size == 0: return None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def are_rois_different(roi1, roi2):
    """Compares two ROIs. Returns True if the pixels have changed (New Floor)."""
    if roi1 is None or roi2 is None: return True
    if roi1.shape != roi2.shape: return True
    # Use template matching correlation to check for identical text
    res = cv2.matchTemplate(roi1, roi2, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)
    # 0.98 is an extremely high bar; any change in a single character will drop the score.
    return score < 0.98 

def run_temporal_chunking():
    if not os.path.exists(DNA_CSV):
        print(f"Error: {DNA_CSV} not found. Run Step 1 DNA sensor first.")
        return

    # Load DNA data. Force DNA columns to strings to prevent int casting issues.
    df = pd.read_csv(DNA_CSV, dtype={'r3_dna': str, 'r4_dna': str, 'dna_sig': str})
    df = df.sort_values('frame_idx').reset_index(drop=True)
    
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 2: KINEMATIC & VISUAL FLOOR GROUPING ---")
    print(f"Processing {len(df)} frames using Game Physics and Visual ROI Verification...")

    # 1. MICRO-BLOCKING
    # A block is defined as a contiguous sequence on the same slot with no large index gaps.
    df['gap'] = df['frame_idx'].diff().fillna(0)
    df['block_id'] = ((df['slot_id'] != df['slot_id'].shift(1)) | (df['gap'] > 5)).cumsum()
    
    blocks = []
    for block_id, group in df.groupby('block_id'):
        blocks.append({
            'start_frame': int(group['frame_idx'].min()),
            'filename': group['filename'].iloc[0],
            'slot': int(group['slot_id'].iloc[0]),
            'r4_mode': str(group['r4_dna'].mode()[0]),
            'r3_mode': str(group['r3_dna'].mode()[0]),
        })
    
    print(f"Phase 1: Consolidated frames into {len(blocks)} slot-attack blocks.")

    # 2. FLOOR GROUPING (Visual & Physical)
    final_floors = []
    last_floor_roi = None
    
    for i, b in enumerate(blocks):
        img = cv2.imread(os.path.join(buffer_dir, b['filename']))
        curr_roi = get_stage_roi(img)
        
        is_new_floor = False
        reason = ""

        if i == 0:
            is_new_floor = True
            reason = "Initial Start"
        else:
            prev_b = blocks[i-1]
            # Rule A: Slot Reversal (Guaranteed floor reset)
            if b['slot'] < prev_b['slot']:
                is_new_floor = True
                reason = f"Slot Reversal ({prev_b['slot']} -> {b['slot']})"
            
            # Rule B: Stage ROI Verification
            # If the header pixels changed, it is a new floor regardless of DNA/Slot.
            # If they are identical, it is the same floor regardless of DNA noise.
            if not is_new_floor:
                if are_rois_different(curr_roi, last_floor_roi):
                    is_new_floor = True
                    reason = "Stage ROI Change (Confirmed Floor Shift)"

        if is_new_floor:
            final_floors.append({
                'floor_id': len(final_floors) + 1,
                'start_frame': b['start_frame'],
                'filename': b['filename'],
                'slot_id': b['slot'],
                'r4_dna': b['r4_mode'],
                'r3_dna': b['r3_mode'],
                'transition_reason': reason
            })
            last_floor_roi = curr_roi

    print(f"Phase 2: Identified {len(final_floors)} distinct floors via Visual Arbiter.")

    # 3. EXPORT & VISUAL PROOFS
    print(f"Exporting Step 2 candidates to: {VERIFY_DIR}")
    for f in final_floors:
        img_path = os.path.join(buffer_dir, f['filename'])
        vis = cv2.imread(img_path)
        if vis is not None:
            h, w = vis.shape[:2]
            # OSD AT BOTTOM (Prevents occluding the Stage ROI header)
            cv2.rectangle(vis, (10, h - 95), (710, h - 10), (0, 0, 0), -1)
            cv2.putText(vis, f"FLOOR {f['floor_id']:03d} | Start Frame: {f['start_frame']}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"DNA: {f['r4_dna']}-{f['r3_dna']} | Slot: {f['slot_id']}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, f"Reason: {f['transition_reason']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            out_name = f"floor_start_{f['floor_id']:03d}_frame_{f['start_frame']}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), vis)

    pd.DataFrame(final_floors).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(final_floors)} validated start frames to: {OUT_CSV}")

if __name__ == "__main__":
    run_temporal_chunking()