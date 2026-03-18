import pandas as pd
import numpy as np
import os
import cv2

# --- CONFIGURATION ---
CSV_PATH = "global_stability_report_v2.csv"
BUFFER_ROOT = "capture_buffer_0"
OUTPUT_DIR = "grand_audit_v543_results"

def run_v5_43_grand_audit():
    # 1. SETUP DIRECTORIES
    image_dir = os.path.join(OUTPUT_DIR, "floor_starts")
    leads_dir = os.path.join(OUTPUT_DIR, "missing_leads")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(leads_dir, exist_ok=True)
    
    print(f"--- Launching v5.43: FIXED Grand Audit Engine ---")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # 2. LOAD & CLEAN DATA
    df = pd.read_csv(CSV_PATH, dtype={'r1_bits': str})
    df['r1_bits'] = df['r1_bits'].fillna('000000').apply(lambda x: str(x).zfill(6))
    df['r1_count'] = df['r1_bits'].apply(lambda x: x.count('1'))
    
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    # --- 3. PEAK ISOLATION ---
    # We filter for frames with signal FIRST, then group them.
    peaks = df[(df['hud_diff'] > 4.5) | (df['hamming'] >= 5)].copy()
    
    if peaks.empty:
        print("Error: No peaks found in CSV. Check your thresholds.")
        return

    # Group the peaks (if gaps between peaks > 25 frames, it's a new floor)
    peaks['event_group'] = (peaks['idx'].diff() > 25).cumsum()
    
    # 4. MANIFEST GENERATION
    # For each peak group, we want the most 'stable' frame at the start
    manifest = peaks.groupby('event_group').agg({
        'idx': 'min',
        'hud_diff': 'max',
        'hamming': 'max',
        'r1_bits': 'first',
        'r1_count': 'min'
    }).reset_index()

    # --- 5. IMAGE EXPORT: DETECTED FLOORS ---
    print(f"Exporting {len(manifest)} detected floor images...")
    for i, row in manifest.iterrows():
        idx = int(row['idx'])
        if idx < len(files):
            img = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
            if img is not None:
                # F1 is Frame 0. So the first detected group is Floor 2.
                out_name = f"Floor_{i+2:03}_Idx_{idx:05}_H{int(row['hamming'])}.jpg"
                cv2.imwrite(os.path.join(image_dir, out_name), img)

    # --- 6. MISSING LEAD DETECTION ---
    # Identify pristine boards (count <= 1) that occur AWAY from detected floors
    detected_indices = set(manifest['idx'].tolist())
    
    # Find candidates
    leads_df = df[(df['r1_count'] <= 1) & (df['hud_diff'] < 1.0)].copy()
    # Filter out Frame 0 (it's always a lead) and frames already in the manifest
    leads_df = leads_df[(leads_df['idx'] > 10) & (~leads_df['idx'].isin(detected_indices))]
    
    if not leads_df.empty:
        missing_groups = leads_df.groupby((leads_df['idx'].diff() > 50).cumsum()).first()
        print(f"Exporting {len(missing_groups)} potential missing floor images...")
        for _, row in missing_groups.iterrows():
            idx = int(row['idx'])
            if idx < len(files):
                img = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
                if img is not None:
                    cv2.imwrite(os.path.join(leads_dir, f"Missing_Lead_Idx_{idx:05}.jpg"), img)

    # --- 7. VELOCITY & REPORTS ---
    manifest['frame_gap'] = manifest['idx'].diff()
    manifest.to_csv(os.path.join(OUTPUT_DIR, "floor_manifest_v543.csv"), index=False)
    
    print(f"\n[GLOBAL AUDIT SUMMARY]")
    print(f"Total Frames:            {len(df)}")
    print(f"Detected Transitions:    {len(manifest)} / 110")
    print(f"Missing Leads Found:     {len(missing_groups) if not leads_df.empty else 0}")
    print(f"Avg Frame Gap:           {manifest['frame_gap'].mean():.1f}")
    
    print(f"\nAudit complete. Check '{OUTPUT_DIR}/floor_starts' for the visual manifest.")

if __name__ == "__main__":
    run_v5_43_grand_audit()