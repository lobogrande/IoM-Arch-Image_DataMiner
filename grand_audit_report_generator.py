import pandas as pd
import numpy as np
import os
import cv2

# --- CONFIGURATION ---
CSV_PATH = "global_stability_report_v2.csv"
BUFFER_ROOT = "capture_buffer_0"
OUTPUT_DIR = "grand_audit_v544_pulse_results"

def run_v5_44_pulse_audit():
    # 1. SETUP DIRECTORIES
    image_dir = os.path.join(OUTPUT_DIR, "floor_starts")
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"--- Launching v5.44: PULSE Grand Audit Engine ---")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # 2. LOAD DATA
    df = pd.read_csv(CSV_PATH, dtype={'r1_bits': str})
    df['r1_bits'] = df['r1_bits'].fillna('000000').apply(lambda x: str(x).zfill(6))
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    # --- 3. PULSE DETECTION LOGIC ---
    # Trigger on a combined spike of HUD change AND Board change
    # Thresholds are tuned to your 10-frame floors (43, 53, 63)
    df['is_spike'] = (df['hud_diff'] > 4.5) & (df['hamming'] >= 4)
    
    # 4. ITERATIVE EXTRACTION (The Pulse Gate)
    # Instead of clustering, we walk the data and enforce a 9-frame lockout
    floor_indices = [0] # Frame 0 is always Floor 1
    last_idx = -100
    
    # Filter for candidates to speed up the loop
    candidates = df[df['is_spike']].to_dict('records')
    
    for cand in candidates:
        idx = int(cand['idx'])
        if idx > last_idx + 9: # The 9-frame "Pulse" Window
            floor_indices.append(idx)
            last_idx = idx

    # --- 5. MANIFEST & IMAGE EXPORT ---
    manifest_data = []
    print(f"Exporting {len(floor_indices)} floor pulse images...")
    
    for i, idx in enumerate(floor_indices):
        if idx < len(files):
            img = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
            if img is not None:
                # Floor 1 = Frame 0, Floor 2 = Frame 43, etc.
                out_name = f"Floor_{i+1:03}_Idx_{idx:05}.jpg"
                cv2.imwrite(os.path.join(image_dir, out_name), img)
                
                # Capture metadata for the CSV
                row = df.iloc[idx]
                manifest_data.append({
                    'floor': i + 1,
                    'idx': idx,
                    'hud_diff': row['hud_diff'],
                    'hamming': row['hamming'],
                    'r1_bits': row['r1_bits']
                })

    # --- 6. OUTPUT ---
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df['frame_gap'] = manifest_df['idx'].diff()
    manifest_df.to_csv(os.path.join(OUTPUT_DIR, "floor_manifest_v544_pulse.csv"), index=False)
    
    print(f"\n[GLOBAL PULSE SUMMARY]")
    print(f"Total Floors Found:      {len(manifest_df)} / 110")
    print(f"Average Floor Life:      {manifest_df['frame_gap'].mean():.1f} frames")
    print(f"Minimum Floor Life:      {manifest_df['frame_gap'].min():.1f} frames")
    
    print(f"\nAudit complete. Visual manifest saved to '{OUTPUT_DIR}/floor_starts'.")

if __name__ == "__main__":
    run_v5_44_pulse_audit()