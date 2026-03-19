import os
import cv2
import pandas as pd
import numpy as np
import shutil

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"  # Path to your 25k images
OUT_BASE = "manual_audit_samples"
GLOBAL_CSV = "global_audit_manifest.csv"
CLEANED_CSV = "cleaned_audit_manifest.csv"

def create_folders():
    for sub in ["discarded", "banners", "multiples"]:
        path = os.path.join(OUT_BASE, sub)
        if not os.path.exists(path): os.makedirs(path)

def draw_overlay(img, row, label, color=(0, 0, 255)):
    h, w, _ = img.shape
    y_top = int(row['y_top'])
    y_bot = y_top + 45 # Assuming our BANNER_H_TARGET
    cv2.rectangle(img, (0, y_top), (w, y_bot), color, 2)
    cv2.putText(img, f"{label} | Y:{y_top}", (20, y_top - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def run_visual_audit():
    # 1. Load Data
    df_global = pd.read_csv(GLOBAL_CSV)
    df_cleaned = pd.read_csv(CLEANED_CSV)
    
    # Identify Discarded frames (In global but not in cleaned)
    # We use frame and y_top as the unique keys
    df_discarded = pd.merge(df_global, df_cleaned, on=['frame', 'y_top'], how='left', indicator=True)
    df_discarded = df_discarded[df_discarded['_merge'] == 'left_only']
    
    # Identify Double Banners
    event_counts = df_cleaned.groupby('frame')['event_id'].nunique()
    double_frames = event_counts[event_counts > 1].index.tolist()
    
    # 2. Get Filenames
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    create_folders()
    print(f"Sampling started. Output folder: {OUT_BASE}")

    # --- CATEGORY 1: Sampling Discarded (Noise) ---
    # We take 20 samples to check if they are indeed stationary HUD/Ores
    discard_samples = df_discarded.sample(min(20, len(df_discarded)))
    for _, row in discard_samples.iterrows():
        fname = all_files[int(row['frame'])]
        img = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img is not None:
            img = draw_overlay(img, row, "REJECTED NOISE", (0, 165, 255)) # Orange
            cv2.imwrite(os.path.join(OUT_BASE, "discarded", f"noise_f{int(row['frame'])}.jpg"), img)

    # --- CATEGORY 2: Sampling True Banners ---
    # We take 2 samples from 10 different random events
    sample_events = df_cleaned['event_id'].unique()
    np.random.shuffle(sample_events)
    for eid in sample_events[:10]:
        event_rows = df_cleaned[df_cleaned['event_id'] == eid].sample(2)
        for _, row in event_rows.iterrows():
            fname = all_files[int(row['frame'])]
            img = cv2.imread(os.path.join(BUFFER_ROOT, fname))
            if img is not None:
                img = draw_overlay(img, row, f"EVENT_{eid}", (0, 255, 0)) # Green
                cv2.imwrite(os.path.join(OUT_BASE, "banners", f"event_{eid}_f{int(row['frame'])}.jpg"), img)

    # --- CATEGORY 3: Multi-Banner Events ---
    # Exporting all frames where > 1 banner exists
    for f_idx in double_frames:
        fname = all_files[int(f_idx)]
        img = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img is not None:
            frame_rows = df_cleaned[df_cleaned['frame'] == f_idx]
            for _, row in frame_rows.iterrows():
                img = draw_overlay(img, row, f"MULTI_EID_{int(row['event_id'])}", (0, 0, 255)) # Red
            cv2.imwrite(os.path.join(OUT_BASE, "multiples", f"double_f{int(f_idx)}.jpg"), img)

    print("Audit Samples Generated. Please inspect the folders in 'manual_audit_samples'.")

if __name__ == "__main__":
    run_visual_audit()