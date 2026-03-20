# dna_sensor_audit.py
# Purpose: Verify the AI's ability to "read" the grid occupancy of Rows 3 and 4 using background templates.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS GRID CONSTANTS (Ore Centers)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# THRESHOLD: If match confidence against "background_plain" is > 0.85, the slot is EMPTY (0).
# Otherwise, it is OCCUPIED (1) by ore, shadow ore, or other sprites.
BG_MATCH_THRESHOLD = 0.85

def get_row_dna(img, row_idx, bg_templates):
    """
    Returns a bitstring (e.g. '101100') for the specified row.
    Occupancy is determined by NOT matching the background template.
    """
    bits = []
    y_center = int(ORE0_Y + (row_idx * STEP))
    
    # ROI Size for background matching (consistent with background_plain templates)
    # Assuming the templates are roughly 40x40 or 30x30 centered on the slot
    tw, th = 30, 30
    
    for col in range(6):
        x_center = int(ORE0_X + (col * STEP))
        
        # Calculate ROI Top-Left
        tx, ty = x_center - (tw // 2), y_center - (th // 2)
        roi = img[ty : ty + th, tx : tx + tw]
        
        if roi.shape[0] < th or roi.shape[1] < tw:
            bits.append('1') # Default to occupied if out of bounds
            continue
            
        # Match against the plain background template
        # We use background_plain_0.png as the primary reference
        res = cv2.matchTemplate(roi, bg_templates[0], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        # LOGIC: High match = Empty (0), Low match = Occupied (1)
        is_empty = max_val >= BG_MATCH_THRESHOLD
        bits.append('0' if is_empty else '1')
        
    return "".join(bits)

def run_dna_audit():
    input_csv = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    if not os.path.exists(input_csv):
        print("Error: sprite_homing_run_0.csv not found. Run Step 1 first.")
        return

    # 1. Load Background Templates
    bg_templates = []
    # Attempt to load up to 3 plain background variants if they exist
    for i in range(3):
        path = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(path):
            bg_templates.append(cv2.imread(path, 0))
    
    if not bg_templates:
        print("Error: No background_plain templates found in templates folder.")
        return

    # Load the "Golden Dataset" from Step 1
    df = pd.read_csv(input_csv)
    source_dir = cfg.get_buffer_path(0)
    
    print(f"--- DNA SENSOR AUDIT (Template-Based) ---")
    print(f"Using Background Matching to detect occupancy in Rows 3 & 4...")
    
    results = []
    if not os.path.exists("dna_debug"): os.makedirs("dna_debug")

    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(source_dir, row['filename']), 0)
        if img is None: continue
        
        # Extract signatures using template logic
        r3_dna = get_row_dna(img, 2, bg_templates) 
        r4_dna = get_row_dna(img, 3, bg_templates)
        combined = f"{r3_dna}-{r4_dna}"
        
        # Visual Verification
        if idx % 50 == 0:
            vis = cv2.imread(os.path.join(source_dir, row['filename']))
            cv2.putText(vis, f"DNA: {combined}", (20, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for r_idx in [2, 3]:
                y = int(ORE0_Y + (r_idx * STEP))
                current_dna = r3_dna if r_idx == 2 else r4_dna
                for c_idx in range(6):
                    x = int(ORE0_X + (c_idx * STEP))
                    # Green = Occupied (1), Red = Empty (0)
                    color = (0, 255, 0) if current_dna[c_idx] == '1' else (0, 0, 255)
                    cv2.circle(vis, (x, y), 6, color, -1)
                    cv2.circle(vis, (x, y), 7, (0,0,0), 1)
            
            cv2.imwrite(f"dna_debug/dna_verify_f{row['frame_idx']}.jpg", vis)

        results.append({
            'frame_idx': row['frame_idx'],
            'r3_dna': r3_dna,
            'r4_dna': r4_dna,
            'dna_sig': combined
        })

    audit_df = pd.DataFrame(results)
    audit_df.to_csv("dna_sensor_results.csv", index=False)
    
    print("\n--- DNA SIGNATURE DISTRIBUTION ---")
    print(f"Unique Signatures: {len(audit_df['dna_sig'].unique())}")
    print(audit_df['dna_sig'].value_counts().head(5))
    print(f"\n[DONE] Check 'dna_debug/' to confirm Green=Occupied, Red=Empty.")

if __name__ == "__main__":
    run_dna_audit()