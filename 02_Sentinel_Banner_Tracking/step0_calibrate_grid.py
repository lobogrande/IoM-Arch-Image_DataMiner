# diag_auto_grid_discovery.py
# Purpose: Automated Brute-Force Scale & Grid discovery to solve the 0.13 confidence crisis.
# Version: 1.0 (Zero-Human Interaction)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_auto_discovery():
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    # Use Frame 0 where the grid is full
    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[0]))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Use Dirt1 as the "Pioneer" template to find the grid
    t_path = os.path.join(cfg.TEMPLATE_DIR, "dirt1_act_plain_0.png")
    tpl_raw = cv2.imread(t_path, 0)
    
    if tpl_raw is None:
        print(f"Error: Could not find dirt1 template at {t_path}")
        return

    print(f"--- STARTING AUTOMATED GEOMETRIC DISCOVERY ---")
    print(f"Analyzing Frame: {all_files[0]}")
    
    best_overall_score = 0
    best_scale = 0
    best_coords = []

    # Sweep through scales: 0.7 to 1.5 (covers standard to 2k/4k variations)
    for scale in np.linspace(0.7, 1.5, 40):
        side = int(48 * scale)
        if side < 10: continue
        tpl_resized = cv2.resize(tpl_raw, (side, side))
        
        # Match against the entire lower 70% of the screen
        search_area = img_gray[int(img_gray.shape[0]*0.2):, :]
        res = cv2.matchTemplate(search_area, tpl_resized, cv2.TM_CCOEFF_NORMED)
        
        # Find the top 12 strongest peaks (enough to prove a row/column)
        locs = np.where(res >= 0.40) # Broad gate
        pts = list(zip(*locs[::-1]))
        
        if not pts: continue

        # Simple deduplication of points too close together
        unique_pts = []
        if pts:
            pts.sort(key=lambda x: res[x[1], x[0]], reverse=True)
            for p in pts:
                if all(np.linalg.norm(np.array(p)-np.array(up)) > side*0.8 for up in unique_pts):
                    unique_pts.append(p)
                if len(unique_pts) >= 12: break
        
        avg_conf = np.mean([res[p[1], p[0]] for p in unique_pts]) if unique_pts else 0
        
        if avg_conf > best_overall_score:
            best_overall_score = avg_conf
            best_scale = scale
            # Adjust y back to global coords
            best_coords = [(p[0] + side//2, p[1] + int(img_gray.shape[0]*0.2) + side//2) for p in unique_pts]

    if not best_coords:
        print("Discovery Failed: No recognizable ore clusters found.")
        return

    print(f"\n[GEOMETRIC TRUTH DISCOVERED]")
    print(f"Peak System Confidence: {best_overall_score:.4f}")
    print(f"Calculated Physical Scale: {best_scale:.3f} ({int(48*best_scale)}px)")
    
    # Deriving Grid Constants from the strongest points found
    best_coords.sort(key=lambda x: (x[1], x[0])) # Sort by Y then X
    
    # Estimate STEP by looking at horizontal distance between adjacent ores
    x_steps = []
    for i in range(len(best_coords)-1):
        dist = abs(best_coords[i+1][0] - best_coords[i][0])
        if 40 < dist < 80: # Reasonable step range
            x_steps.append(dist)
    
    final_step = np.median(x_steps) if x_steps else 59.0
    
    # Anchor is the top-left-most point found
    anchor = best_coords[0]
    
    print(f"Suggested ORE0_X = {anchor[0]}")
    print(f"Suggested ORE0_Y = {anchor[1]}")
    print(f"Suggested STEP = {final_step}")
    
    # Final Visual Validation
    vis = img_bgr.copy()
    for p in best_coords:
        cv2.circle(vis, p, 5, (0, 255, 0), -1)
        cv2.rectangle(vis, (p[0]-int(24*best_scale), p[1]-int(24*best_scale)), 
                           (p[0]+int(24*best_scale), p[1]+int(24*best_scale)), (255, 0, 255), 1)
    
    cv2.imwrite("auto_grid_discovery_check.jpg", vis)
    print("\nVisual check saved: 'auto_grid_discovery_check.jpg'.")
    print("If the purple boxes are perfectly centered, copy the suggested constants.")

if __name__ == "__main__":
    run_auto_discovery()