# sprite_right_side_profiler.py
# Version: 1.0
# Purpose: Profile all potential player matches on the right side (Slots 4 & 5).

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIG ---
LIMIT = 10000
# Scanning a wider horizontal strip to catch any drift
Y_STRIP = (220, 290) 
X_RANGE = (220, 480) # Covers from Slot 3.5 to the end of Row 1

def run_right_side_profile():
    print(f"--- RIGHT-SIDE SPRITE PROFILER (0-{LIMIT}) ---")
    
    tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    if tpl is None: return
    th, tw = tpl.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:LIMIT]
    
    profile_data = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        # Focus on the right-side strip of Row 1
        strip = img[Y_STRIP[0]:Y_STRIP[1], X_RANGE[0]:X_RANGE[1]]
        
        # We look for the absolute best match in this region
        res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Log any peak above 0.50 (lowered to see the "noise")
        global_x = max_loc[0] + X_RANGE[0]
        global_y = max_loc[1] + Y_STRIP[0]

        profile_data.append({
            'frame': f_idx,
            'peak_x': global_x,
            'peak_y': global_y,
            'confidence': round(max_val, 4)
        })

        if f_idx % 2000 == 0:
            print(f"  Processed {f_idx} frames...")

    # Save Data
    df = pd.DataFrame(profile_data)
    df.to_csv("right_side_profile.csv", index=False)
    
    # Generate Visualization: X vs Confidence
    plt.figure(figsize=(12, 6))
    plt.scatter(df['peak_x'], df['confidence'], alpha=0.3, c=df['confidence'], cmap='viridis')
    
    # Draw Vertical Lines for where we THINK S4 and S5 are
    # S0_X=11 + (4*59) = 247 | S0_X=11 + (5*59) = 306
    plt.axvline(x=247, color='red', linestyle='--', label='Expected S4 Stand (X=247)')
    plt.axvline(x=306, color='blue', linestyle='--', label='Expected S5 Stand (X=306)')
    
    plt.title(f"Right-Side Detection Profile (Frames 0-{LIMIT})")
    plt.xlabel("Detected X-Coordinate (Top-Left)")
    plt.ylabel("Match Confidence")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("right_side_signal_map.png")
    
    print("\n[DONE] Profile complete.")
    print("Check 'right_side_signal_map.png'. Are there clusters near the lines?")

if __name__ == "__main__":
    run_right_side_profile()