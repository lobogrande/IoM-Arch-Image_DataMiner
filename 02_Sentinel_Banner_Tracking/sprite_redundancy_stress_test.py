# sprite_redundancy_stress_test.py
# Purpose: Prove that Max(Full, Bottom) hybrid logic stabilizes Slot 2-5 detections.

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS CONSTANTS
AI_S0 = (11, 225)
STEP = 59.0
FLIP = 82.0
LIMIT = 2000 # Focusing on the first transition and center-board mining

def run_stress_test():
    print(f"--- HYBRID REDUNDANCY STRESS TEST (0-{LIMIT}) ---")
    
    # 1. Load All Templates
    tpl_r_full = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_r_bot  = tpl_r_full[30:, :]
    tpl_l_full = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    tpl_l_bot  = tpl_l_full[30:, :]
    
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:LIMIT]
    results = []

    # Slots to monitor (The most problematic ones)
    monitor_slots = [2, 3, 4, 5]

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        row_data = {'frame': f_idx}
        for s_id in monitor_slots:
            # Physics
            col = s_id % 6
            x_tl, y_tl = int(AI_S0[0] + (col * STEP)), int(AI_S0[1])
            
            # 1. Full-Body Match
            roi_f = img[y_tl : y_tl+60, x_tl : x_tl+40]
            score_f = cv2.minMaxLoc(cv2.matchTemplate(roi_f, tpl_r_full, cv2.TM_CCOEFF_NORMED))[1] if roi_f.shape == (60,40) else 0
            
            # 2. Bottom-Half Match
            roi_b = img[y_tl+30 : y_tl+60, x_tl : x_tl+40]
            score_b = cv2.minMaxLoc(cv2.matchTemplate(roi_b, tpl_r_bot, cv2.TM_CCOEFF_NORMED))[1] if roi_b.shape == (30,40) else 0

            # 3. The Hybrid Winner
            row_data[f's{s_id}_hybrid'] = max(score_f, score_b)
        
        results.append(row_data)

    # 2. Visualization
    df = pd.DataFrame(results)
    plt.figure(figsize=(15, 6))
    for s_id in monitor_slots:
        plt.plot(df['frame'], df[f's{s_id}_hybrid'], label=f"Slot {s_id} (Hybrid)", alpha=0.7)
    
    plt.axhline(y=0.76, color='red', linestyle='--', label='Proposed Stability Floor')
    plt.title("Hybrid Redundancy Signal (Slots 2, 3, 4, 5)")
    plt.ylabel("Confidence (Max of Full or Bottom)")
    plt.legend()
    plt.savefig("hybrid_redundancy_signal.png")
    
    print("\n[DONE] Check 'hybrid_redundancy_signal.png'.")
    print("Does the hybrid signal stay above 0.76 during mining pulses?")

if __name__ == "__main__":
    run_stress_test()