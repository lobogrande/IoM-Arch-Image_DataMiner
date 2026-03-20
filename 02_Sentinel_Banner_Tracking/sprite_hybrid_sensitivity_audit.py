# sprite_hybrid_sensitivity_audit.py
# Purpose: Compare Full-Body vs Bottom-Half scores across all 7 slots.

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS CONSTANTS (AI TOP-LEFT)
S0_X, S0_Y = 11, 225
STEP = 59.0
LIMIT = 5000

def run_hybrid_audit():
    print(f"--- HYBRID SENSITIVITY AUDIT (0-{LIMIT}) ---")
    
    # 1. Load Templates
    tpl_r_full = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l_full = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r_full is None: return
    
    th, tw = tpl_r_full.shape
    tpl_r_bot = tpl_r_full[30:, :]
    tpl_l_bot = tpl_l_full[30:, :]
    bh, bw = tpl_r_bot.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:LIMIT]
    
    # Define Slots
    slots = []
    for s in range(6):
        slots.append({'id': s, 'x': int(S0_X + (s * STEP)), 'y': int(S0_Y), 'tpl_f': tpl_r_full, 'tpl_b': tpl_r_bot})
    slots.append({'id': 11, 'x': int(S0_X + (5 * STEP)), 'y': int(S0_Y + STEP), 'tpl_f': tpl_l_full, 'tpl_b': tpl_l_bot})

    audit_results = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        frame_data = {'frame': f_idx}
        for s in slots:
            # Full Body ROI
            roi_f = img[s['y']:s['y']+th, s['x']:s['x']+tw]
            # Bottom Half ROI (Offset by 30px)
            roi_b = img[s['y']+30:s['y']+30+bh, s['x']:s['x']+bw]

            def match(roi, tpl):
                if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]: return 0
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                return cv2.minMaxLoc(res)[1]

            frame_data[f"s{s['id']}_full"] = match(roi_f, s['tpl_f'])
            frame_data[f"s{s['id']}_bot"] = match(roi_b, s['tpl_b'])
        
        audit_results.append(frame_data)
        if f_idx % 1000 == 0: print(f"  Processed {f_idx} frames...")

    # 2. Save and Report
    df = pd.DataFrame(audit_results)
    df.to_csv("hybrid_sensitivity_report.csv", index=False)
    
    # Summary Table for the Terminal
    print("\n--- MEAN CONFIDENCE PER SLOT ---")
    summary = []
    for s in slots:
        f_col, b_col = f"s{s['id']}_full", f"s{s['id']}_bot"
        # We only look at frames where there is a likely hit (>0.60) to avoid averaging noise
        f_mean = df[df[f_col] > 0.6][f_col].mean()
        b_mean = df[df[b_col] > 0.6][b_col].mean()
        summary.append({'Slot': s['id'], 'Full_Body_Avg': round(f_mean, 3), 'Bottom_Half_Avg': round(b_mean, 3)})
    
    print(pd.DataFrame(summary))
    print("\n[DONE] Check 'hybrid_sensitivity_report.csv' for the raw frame-by-frame delta.")

if __name__ == "__main__":
    run_hybrid_audit()