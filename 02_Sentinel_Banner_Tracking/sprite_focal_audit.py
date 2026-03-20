# sprite_focal_audit.py
# Purpose: Pinpoint the exact pixel coordinates of the player in known frames.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_focal_audit():
    # Targeted frames where we know the player's slot position
    test_frames = {
        63:  "Expected: Slot 0 (Right)",
        120: "Expected: Slot 1 (Right)",
        213: "Expected: Slot 11 (Left)", # Transition to Floor 13
        240: "Expected: Slot 2 (Right)",
        550: "Expected: Slot 3 (Right)"
    }
    
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    # 1. Load Templates
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r is None or tpl_l is None: return
    th, tw = tpl_r.shape

    results = []
    print(f"--- FOCAL SPRITE AUDIT (5 Key Frames) ---")

    for f_idx, label in test_frames.items():
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]), 0)
        if img is None: continue
        ih, iw = img.shape
        
        # Scan Row 1 (y: 200-300) and Row 2 (y: 270-370)
        # We search the FULL width to find the true peak
        r1_strip = img[200:300, :]
        r2_strip = img[270:370, :]
        
        # Peak Match for Right Facing
        res_r = cv2.matchTemplate(r1_strip, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
        
        # Peak Match for Left Facing
        res_l = cv2.matchTemplate(r2_strip, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
        
        results.append({
            'frame': f_idx,
            'label': label,
            'best_r_x': loc_r[0],
            'score_r': round(val_r, 4),
            'best_l_x': loc_l[0],
            'score_l': round(val_l, 4)
        })
        print(f"Frame {f_idx}: Peak Right X={loc_r[0]} (Conf:{round(val_r,3)}) | Peak Left X={loc_l[0]} (Conf:{round(val_l,3)})")

    # 2. Comparison to our 59px Grid
    df = pd.DataFrame(results)
    df.to_csv("sprite_focal_audit_results.csv", index=False)
    print("\n[DONE] CSV saved. Please provide the 'best_r_x' and 'best_l_x' values for these 5 frames.")

if __name__ == "__main__":
    run_focal_audit()