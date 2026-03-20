# dna_sensor_audit.py
# Purpose: Verify the AI's ability to "read" the grid occupancy of Rows 3 and 4.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS GRID CONSTANTS (Ore Centers)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# THRESHOLD: Average intensity below which a slot is considered "Empty/Mined"
# This is the key value we are auditing.
EMPTY_THRESHOLD = 45 

def get_row_dna(img, row_idx):
    """
    Returns a bitstring (e.g. '101100') for the specified row.
    row_idx 0 = Row 1, row_idx 1 = Row 2, etc.
    We are auditing Row 3 (idx 2) and Row 4 (idx 3).
    """
    bits = []
    # Calculate Y center for the specific row
    y_center = int(ORE0_Y + (row_idx * STEP))
    
    for col in range(6):
        # Calculate X center for the specific column
        x_center = int(ORE0_X + (col * STEP))
        
        # Sample a 10x10 patch at the center of the ore slot
        # We use a patch rather than a single pixel to average out noise/sparkles
        patch = img[y_center-5:y_center+5, x_center-5:x_center+5]
        avg_brightness = np.mean(patch)
        
        # 1 = Ore Present, 0 = Empty Space/Mined
        bits.append('1' if avg_brightness > EMPTY_THRESHOLD else '0')
        
    return "".join(bits)

def run_dna_audit():
    input_csv = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    if not os.path.exists(input_csv):
        print("Error: sprite_homing_run_0.csv not found. Run Step 1 first.")
        return

    # Load the "Golden Dataset" from Step 1
    df = pd.read_csv(input_csv)
    source_dir = cfg.get_buffer_path(0)
    
    print(f"--- DNA SENSOR AUDIT ---")
    print(f"Auditing Row 3 & 4 signatures for {len(df)} detected frames...")
    
    results = []
    if not os.path.exists("dna_debug"): os.makedirs("dna_debug")

    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(source_dir, row['filename']), 0)
        if img is None: continue
        
        # Extract signatures
        r3_dna = get_row_dna(img, 2) # Row 3 (Index 2)
        r4_dna = get_row_dna(img, 3) # Row 4 (Index 3)
        combined = f"{r3_dna}-{r4_dna}"
        
        # Visual Verification: Save an image periodically or for specific changes
        # For this audit, we'll save every 50 frames to see a variety of floors
        if idx % 50 == 0:
            vis = cv2.imread(os.path.join(source_dir, row['filename']))
            
            # Label the frame with the detected DNA
            cv2.putText(vis, f"DNA: {combined}", (20, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"Frame: {row['frame_idx']} | Slot: {row['slot_id']}", (20, 435), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw sample points on the image for visual confirmation
            for r_idx in [2, 3]:
                y = int(ORE0_Y + (r_idx * STEP))
                current_dna = r3_dna if r_idx == 2 else r4_dna
                for c_idx in range(6):
                    x = int(ORE0_X + (c_idx * STEP))
                    # Green = Detected Ore (1), Red = Empty (0)
                    color = (0, 255, 0) if current_dna[c_idx] == '1' else (0, 0, 255)
                    cv2.circle(vis, (x, y), 6, color, -1)
                    cv2.circle(vis, (x, y), 7, (0,0,0), 1) # Black outline
            
            cv2.imwrite(f"dna_debug/dna_verify_f{row['frame_idx']}.jpg", vis)

        results.append({
            'frame_idx': row['frame_idx'],
            'r3_dna': r3_dna,
            'r4_dna': r4_dna,
            'dna_sig': combined
        })

    audit_df = pd.DataFrame(results)
    
    # Statistical Summary
    print("\n--- DNA SIGNATURE DISTRIBUTION ---")
    sig_counts = audit_df['dna_sig'].value_counts()
    print(f"Unique Signatures Found: {len(sig_counts)}")
    print("\nTop 5 Most Frequent Signatures (Floor Candidates):")
    print(sig_counts.head(5))
    
    # Save the audit results
    audit_df.to_csv("dna_sensor_results.csv", index=False)
    print(f"\n[DONE] Results saved to 'dna_sensor_results.csv'.")
    print("Check 'dna_debug/' to confirm that Green/Red dots match the ore slots.")

if __name__ == "__main__":
    run_dna_audit()