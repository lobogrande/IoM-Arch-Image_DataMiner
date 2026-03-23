# step0_calibrate_grid.py
# Purpose: Automated Brute-Force Scale & Grid discovery for new datasets.
# Version: 1.1 (Standardized Output & Row-Correction)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_auto_discovery():
    # SET THIS TO THE BUFFER YOU ARE TESTING
    CURRENT_BUFFER = 1 
    
    buffer_dir = cfg.get_buffer_path(CURRENT_BUFFER)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[0]))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    t_path = os.path.join(cfg.TEMPLATE_DIR, "dirt1_act_plain_0.png")
    tpl_raw = cv2.imread(t_path, 0)
    
    if tpl_raw is None:
        print(f"Error: Could not find dirt1 template at {t_path}")
        return

    print(f"--- STEP 0: AUTOMATED GRID DISCOVERY (RUN {CURRENT_BUFFER}) ---")
    
    best_overall_score = 0
    best_scale = 0
    best_coords = []

    for scale in np.linspace(0.8, 1.2, 20):
        side = int(48 * scale)
        tpl_resized = cv2.resize(tpl_raw, (side, side))
        search_area = img_gray[int(img_gray.shape[0]*0.2):, :]
        res = cv2.matchTemplate(search_area, tpl_resized, cv2.TM_CCOEFF_NORMED)
        
        locs = np.where(res >= 0.45)
        pts = list(zip(*locs[::-1]))
        if not pts: continue

        unique_pts = []
        pts.sort(key=lambda x: res[x[1], x[0]], reverse=True)
        for p in pts:
            if all(np.linalg.norm(np.array(p)-np.array(up)) > side*0.8 for up in unique_pts):
                unique_pts.append(p)
            if len(unique_pts) >= 12: break
        
        avg_conf = np.mean([res[p[1], p[0]] for p in unique_pts]) if unique_pts else 0
        if avg_conf > best_overall_score:
            best_overall_score = avg_conf
            best_scale = scale
            best_coords = [(p[0] + side//2, p[1] + int(img_gray.shape[0]*0.2) + side//2) for p in unique_pts]

    if not best_coords:
        print("Discovery Failed.")
        return

    best_coords.sort(key=lambda x: (x[1], x[0])) 
    x_steps = []
    for i in range(len(best_coords)-1):
        dist = abs(best_coords[i+1][0] - best_coords[i][0])
        if 45 < dist < 75: x_steps.append(dist)
    final_step = np.median(x_steps) if x_steps else 59.0
    
    anchor = best_coords[0]
    # Row Correction: If Y is significantly below 261, it found Row 2.
    suggested_y = anchor[1]
    if suggested_y > 300:
        print(f"[!] Warning: First detection at Y={suggested_y} (Likely Row 2). Correcting to Row 1 baseline...")
        suggested_y -= final_step

    print(f"\n[GEOMETRIC TRUTH DISCOVERED]")
    print(f"Calculated Scale: {best_scale:.3f}")
    print(f"Suggested ORE0_X = {anchor[0]}")
    print(f"Suggested ORE0_Y = {int(suggested_y)}")
    print(f"Suggested STEP   = {final_step}")
    
    # Save to Calibration Vault
    vis = img_bgr.copy()
    for p in best_coords:
        cv2.circle(vis, p, 5, (0, 255, 0), -1)
        cv2.rectangle(vis, (p[0]-24, p[1]-24), (p[0]+24, p[1]+24), (255, 0, 255), 1)
    
    out_path = os.path.join(cfg.DATA_DIRS["CALIB"], f"run_{CURRENT_BUFFER}_grid_discovery.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\nVisual check saved to Calibration Vault: {out_path}")

if __name__ == "__main__":
    run_auto_discovery()