# step2_dna_occupancy.py
# Purpose: Generate the final DNA occupancy map for all Step 1 frames.
# Version: 2.0 (Dynamic Pathing & Validated Constants)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]

INPUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{RUN_ID}.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"dna_sensor_run_{RUN_ID}.csv")

# --- VALIDATED GRID CONSTANTS (Ore Centers) ---
ORE0_X, ORE0_Y = 74, 261
STEP = 59.0

# REFINED THRESHOLD: Based on profiling, "Empty" scores are > 0.90 
# and "Occupied" are < 0.40. 0.75 is a safe mid-point.
VALLEY_THRESHOLD = 0.75

def load_all_bg_templates():
    templates =[]
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(p):
            templates.append({'img': cv2.imread(p, 0)})
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"negative_ui_{i}.png")
        if os.path.exists(p):
            templates.append({'img': cv2.imread(p, 0)})
    return templates

def get_slot_dna(img, row_idx, col_idx, templates):
    """Returns a bit (0=Empty, 1=Occupied) and the max confidence score."""
    y_center = int(ORE0_Y + (row_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    
    tw, th = 30, 30
    tx, ty = x_center - (tw // 2), y_center - (th // 2)
    roi = img[ty : ty + th, tx : tx + tw]
    
    if roi.shape[0] < th or roi.shape[1] < tw:
        return '1', 0.0 # Boundary failure defaults to occupied
        
    best_val = -1
    for t in templates:
        res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            
    # Logic: High match with background = Empty (0)
    bit = '0' if best_val >= VALLEY_THRESHOLD else '1'
    return bit, round(float(best_val), 4)

def run_final_dna_scan():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Ensure Step 1 has completed.")
        return

    templates = load_all_bg_templates()
    df = pd.read_csv(INPUT_CSV)
    
    print(f"--- STEP 2: DNA OCCUPANCY SCAN (Target: Run {RUN_ID} | Frames: {len(df)}) ---")
    
    results =[]
    ambiguous_count = 0

    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(SOURCE_DIR, row['filename']), 0)
        if img is None: continue
        
        r3_bits, r4_bits = [],[]
        scores = []
        
        # Process Rows 3 and 4
        for r_idx in[2, 3]:
            for c_idx in range(6):
                bit, score = get_slot_dna(img, r_idx, c_idx, templates)
                scores.append(score)
                if r_idx == 2: r3_bits.append(bit)
                else: r4_bits.append(bit)
                
                # Separation Check: Log if a score lands in the "valley"
                if 0.50 < score < 0.70:
                    ambiguous_count += 1

        r3_dna = "".join(r3_bits)
        r4_dna = "".join(r4_bits)
        
        results.append({
            'frame_idx': row['frame_idx'],
            'filename': row['filename'],
            'slot_id': row['slot_id'],
            'r3_dna': r3_dna,
            'r4_dna': r4_dna,
            'dna_sig': f"{r3_dna}-{r4_dna}",
            'min_score': min(scores),
            'max_score': max(scores)
        })

        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} frames...")

    # Save final dataset
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUT_CSV, index=False)
    
    print(f"\n--- SCAN COMPLETE ---")
    print(f"Saved: {os.path.basename(OUT_CSV)}")
    print(f"Ambiguous Scores Detected: {ambiguous_count} (Lower is better)")
    print(f"Unique DNA Signatures: {len(final_df['dna_sig'].unique())}")
    
    # Show the "Signature Cliff" - How many frames share the same DNA?
    print("\nTop 10 DNA Signatures by Frame Count:")
    print(final_df['dna_sig'].value_counts().head(10))

if __name__ == "__main__":
    run_final_dna_scan()