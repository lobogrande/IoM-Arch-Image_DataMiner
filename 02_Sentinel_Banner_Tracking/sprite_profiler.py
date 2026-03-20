# sprite_profiler.py
# Version: 1.0 (Profiling Experiment)
# Purpose: Log peak confidence and X-coordinates for every frame to diagnose grid drift.

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- EXPERIMENT CONFIG ---
TARGET_FRAMES = 2000
ROW1_Y = 249  # Calibrated Row 1 Y
STEP_X = 118.0 
ANCHOR_X = 11.0
OUT_CSV = "sprite_profile_data.csv"
OUT_PLOT = "sprite_profile_signal.png"

def run_profiler():
    print(f"--- SPRITE PROFILING EXPERIMENT (First {TARGET_FRAMES} Frames) ---")
    
    # 1. Load Template (Full sprite)
    tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    if tpl is None: return
    th, tw = tpl.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:TARGET_FRAMES]
    
    profile_data = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue
        ih, iw = img.shape

        # Define the scan strip for Row 1
        y1, y2 = max(0, ROW1_Y - 40), min(ih, ROW1_Y + 40)
        strip = img[y1:y2, :]

        # Template Match across the FULL width
        res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Signal-to-Noise Ratio (SNR)
        # Ratio of the peak to the median background noise in the strip
        snr = max_val / (np.median(res) + 1e-6)

        # Inferred Slot based on our 118px grid
        # found_x (sprite center) = max_loc[0] + half_width + wait_offset(55)
        found_center_x = max_loc[0] + (tw // 2) + 55
        inferred_slot = (found_center_x - ANCHOR_X) / STEP_X

        profile_data.append({
            'frame': f_idx,
            'confidence': max_val,
            'peak_x': max_loc[0],
            'snr': round(snr, 2),
            'slot_float': round(inferred_slot, 2)
        })

        if f_idx % 500 == 0:
            print(f"  Profiling: {f_idx}/{TARGET_FRAMES} frames...")

    # 2. Save Data
    df = pd.DataFrame(profile_data)
    df.to_csv(OUT_CSV, index=False)

    # 3. Generate Signal Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Subplot 1: Confidence over Time
    ax1.plot(df['frame'], df['confidence'], color='blue', alpha=0.6, label='Peak Confidence')
    ax1.axhline(y=0.82, color='red', linestyle='--', label='Current Threshold (0.82)')
    ax1.set_ylabel("Confidence Score")
    ax1.set_title(f"Sprite Signal Strength Profile (Frames 0-{TARGET_FRAMES})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Peak X-Coordinate (The Staircase)
    # This will show us if the player is actually moving right or staying at 0
    ax2.scatter(df['frame'], df['peak_x'], c=df['confidence'], cmap='viridis', s=10)
    ax2.set_ylabel("Detected X-Pixel")
    ax2.set_xlabel("Frame Index")
    ax2.set_title("Detected Player Movement (X-Coordinate vs Time)")
    plt.colorbar(ax2.collections[0], ax=ax2, label='Confidence')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    print(f"\n[DONE] Profiling complete. CSV: {OUT_CSV}, Plot: {OUT_PLOT}")

if __name__ == "__main__":
    run_profiler()