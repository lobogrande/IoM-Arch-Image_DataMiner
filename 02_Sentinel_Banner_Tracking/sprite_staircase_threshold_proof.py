# sprite_staircase_threshold_proof.py
# Version: 1.0
# Purpose: Prove that per-slot staircase thresholds capture the "missing" right-side events.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- THE FINAL PHYSICS ---
AI_S0_X, AI_S0_Y = 11, 225  # AI Top-Left for Slot 0
STEP = 59.0
STAND_FLIP = 82.0           # Distance from Stand-Left to Stand-Right (41 * 2)
LIMIT = 5000                # Testing on the first 5000 frames

# --- THE STAIRCASE THRESHOLDS ---
STAIRCASE = {
    0: 0.90,  # Pristine (Left)
    1: 0.85,
    2: 0.82,
    3: 0.78,
    4: 0.75,
    5: 0.72,  # Maximum Sensitivity (Right/Noisy edge)
    11: 0.82  # Clean Row 2 (Facing Left)
}

def run_staircase_proof():
    print(f"--- STAIRCASE THRESHOLD PROOF (Frames 0-{LIMIT}) ---")
    
    # 1. Load and Prepare Templates
    full_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    full_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if full_r is None or full_l is None:
        print("Error: Player templates not found.")
        return
        
    th, tw = full_r.shape # 60x40
    bot_r = full_r[30:, :] # Bottom 30px
    bot_l = full_l[30:, :]
    bh, bw = bot_r.shape   # 30x40

    buffer_path = cfg.get_buffer_path(0)
    files = sorted([f for f in os.listdir(buffer_path) if f.endswith('.png')])[:LIMIT]
    
    results = []
    counts = Counter()

    # Define targets for Slots 0-5 and 11
    targets = []
    for s_id in list(range(6)) + [11]:
        col = s_id % 6
        row = 0 if s_id < 6 else 1
        
        # Calculate AI Top-Left Base
        x_tl = int(AI_S0_X + (col * STEP))
        y_tl = int(AI_S0_Y + (row * STEP))
        
        # Apply Stand-Flip for Slot 11 (Facing Left)
        if s_id == 11:
            x_tl += int(STAND_FLIP)
            tpl_f, tpl_b = full_l, bot_l
        else:
            tpl_f, tpl_b = full_r, bot_r
            
        targets.append({
            'slot': s_id, 
            'x': x_tl, 
            'y': y_tl, 
            'tpl_f': tpl_f, 
            'tpl_b': tpl_b, 
            'thresh': STAIRCASE[s_id]
        })

    print(f"Scanning using hybrid max(Full, Bottom) matching...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(buffer_path, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for t in targets:
            # 1. Full-Body ROI
            roi_f = img[t['y'] : t['y']+th, t['x'] : t['x']+tw]
            score_f = 0
            if roi_f.shape == (th, tw):
                score_f = cv2.minMaxLoc(cv2.matchTemplate(roi_f, t['tpl_f'], cv2.TM_CCOEFF_NORMED))[1]
            
            # 2. Bottom-Half ROI (Offset by 30px)
            roi_b = img[t['y']+30 : t['y']+30+bh, t['x'] : t['x']+bw]
            score_b = 0
            if roi_b.shape == (bh, bw):
                score_b = cv2.minMaxLoc(cv2.matchTemplate(roi_b, t['tpl_b'], cv2.TM_CCOEFF_NORMED))[1]

            # 3. Hybrid Winner
            best_score = max(score_f, score_b)

            if best_score >= t['thresh']:
                results.append({
                    'frame': f_idx,
                    'slot': t['slot'],
                    'conf': round(best_score, 4),
                    'method': 'Full' if score_f >= score_b else 'Bottom'
                })
                counts[t['slot']] += 1
                break # Only one detection per frame

        if f_idx % 1000 == 0:
            print(f"  Processed {f_idx} frames...")

    # 4. Output Summary
    print("\n--- DETECTION SUMMARY (STAIRCASE MODEL) ---")
    summary_df = pd.DataFrame([
        {'Slot': s, 'Threshold': STAIRCASE[s], 'Hits': counts[s]} 
        for s in sorted(STAIRCASE.keys())
    ])
    print(summary_df)
    
    # Save the hits to a verification CSV
    pd.DataFrame(results).to_csv("staircase_proof_results.csv", index=False)
    print("\n[DONE] Results saved to 'staircase_proof_results.csv'.")

if __name__ == "__main__":
    run_staircase_proof()