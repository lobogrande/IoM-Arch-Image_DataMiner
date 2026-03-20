# sprite_right_side_pulse_audit.py
# Purpose: Prove that the player is detectable in Slots 4 & 5 using bottom-weighted matching.

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS COORDINATES (AI TOP-LEFT)
S0_X, S0_Y = 11, 225
STEP = 59.0
LIMIT = 10000

def run_pulse_audit():
    print(f"--- RIGHT-SIDE PULSE AUDIT (S4 & S5 | Frames 0-{LIMIT}) ---")
    
    # 1. Load and Prepare Templates
    raw_tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    if raw_tpl is None: return
    th, tw = raw_tpl.shape
    
    # Bottom-Half Template (Ignoring the head/UI area)
    bottom_tpl = raw_tpl[int(th*0.5):, :]
    bth, btw = bottom_tpl.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:LIMIT]
    
    # Trackers for S4 and S5
    s4_x, s5_x = int(S0_X + (4 * STEP)), int(S0_X + (5 * STEP))
    results = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        # Slot 4 Check
        roi4 = img[S0_Y : S0_Y+th, s4_x : s4_x+tw]
        # Slot 5 Check
        roi5 = img[S0_Y : S0_Y+th, s5_x : s5_x+tw]

        def get_score(roi, template):
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]: return 0
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            return val

        results.append({
            'frame': f_idx,
            's4_full': get_score(roi4, raw_tpl),
            's4_bottom': get_score(roi4[int(th*0.5):, :], bottom_tpl),
            's5_full': get_score(roi5, raw_tpl),
            's5_bottom': get_score(roi5[int(th*0.5):, :], bottom_tpl)
        })

        if f_idx % 2000 == 0:
            print(f"  Processed {f_idx} frames...")

    # 2. Plotting the "Pulses"
    df = pd.DataFrame(results)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax1.plot(df['frame'], df['s4_full'], label='Full Body', alpha=0.4, color='gray')
    ax1.plot(df['frame'], df['s4_bottom'], label='Bottom Half (Feet)', color='green')
    ax1.set_title("Slot 4 Pulse Audit (Expected AI X=247)")
    ax1.legend()
    
    ax2.plot(df['frame'], df['s5_full'], label='Full Body', alpha=0.4, color='gray')
    ax2.plot(df['frame'], df['s5_bottom'], label='Bottom Half (Feet)', color='blue')
    ax2.set_title("Slot 5 Pulse Audit (Expected AI X=306)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("right_side_pulse_comparison.png")
    print("\n[DONE] Plot saved: 'right_side_pulse_comparison.png'")

if __name__ == "__main__":
    run_pulse_audit()