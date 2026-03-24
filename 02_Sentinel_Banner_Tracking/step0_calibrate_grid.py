# step0_calibrate_grid.py
# Purpose: Automated Brute-Force Scale & Grid discovery for new datasets.
# Version: 1.2 (Dynamic Pathing & Baseline Interpretation)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- ESTABLISHED BASELINE ---
# These are the expected constants for a perfectly aligned 1080p Lava Biome run
BASELINE_X = 74
BASELINE_Y = 261
BASELINE_STEP = 59.0

def run_auto_discovery():
    # --- DYNAMIC CONFIGURATION ---
    buffer_dir = cfg.get_buffer_path()
    run_id = os.path.basename(buffer_dir).split('_')[-1]
    
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    if not all_files:
        print(f"Error: No images found in {buffer_dir}")
        return
        
    # Use Frame 0 where the grid is full
    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[0]))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Use Dirt1 as the "Pioneer" template to find the grid
    t_path = os.path.join(cfg.TEMPLATE_DIR, "dirt1_act_plain_0.png")
    tpl_raw = cv2.imread(t_path, 0)
    
    if tpl_raw is None:
        print(f"Error: Could not find dirt1 template at {t_path}")
        return

    print(f"--- STEP 0: AUTOMATED GRID DISCOVERY (RUN {run_id}) ---")
    print(f"Analyzing Frame: {all_files[0]}")
    
    best_overall_score = 0
    best_scale = 0
    best_coords =[]

    # Sweep through scales: 0.8 to 1.2
    for scale in np.linspace(0.8, 1.2, 20):
        side = int(48 * scale)
        if side < 10: continue
        tpl_resized = cv2.resize(tpl_raw, (side, side))
        
        # Match against the entire lower 80% of the screen
        search_area = img_gray[int(img_gray.shape[0]*0.2):, :]
        res = cv2.matchTemplate(search_area, tpl_resized, cv2.TM_CCOEFF_NORMED)
        
        # Find the top strongest peaks (enough to prove a row/column)
        locs = np.where(res >= 0.45) 
        pts = list(zip(*locs[::-1]))
        
        if not pts: continue

        # Simple deduplication of points too close together
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
            # Adjust y back to global coords
            best_coords = [(p[0] + side//2, p[1] + int(img_gray.shape[0]*0.2) + side//2) for p in unique_pts]

    if not best_coords:
        print("Discovery Failed: No recognizable ore clusters found.")
        return

    # Deriving Grid Constants from the strongest points found
    best_coords.sort(key=lambda x: (x[1], x[0])) # Sort by Y then X
    
    # Estimate STEP by looking at horizontal distance between adjacent ores
    x_steps =[]
    for i in range(len(best_coords)-1):
        dist = abs(best_coords[i+1][0] - best_coords[i][0])
        if 45 < dist < 75: # Reasonable step range
            x_steps.append(dist)
    
    final_step = np.median(x_steps) if x_steps else BASELINE_STEP
    
    # Anchor is the top-left-most point found
    anchor = best_coords[0]
    
    # Row Correction: If Y is significantly below the baseline, it probably found Row 2
    suggested_y = anchor[1]
    if suggested_y > (BASELINE_Y + (final_step * 0.8)):
        print(f"[!] Warning: First detection at Y={suggested_y} (Likely Row 2). Correcting to Row 1 baseline...")
        # If it found slot 6 instead of slot 1, we subtract one full row step
        suggested_y -= final_step

    suggested_x = int(anchor[0])
    suggested_y = int(suggested_y)
    suggested_step = round(final_step, 1)

    print(f"\n[GEOMETRIC TRUTH DISCOVERED]")
    print(f"Calculated Scale: {best_scale:.3f}")
    print(f"Suggested ORE0_X = {suggested_x} (Baseline: {BASELINE_X})")
    print(f"Suggested ORE0_Y = {suggested_y} (Baseline: {BASELINE_Y})")
    print(f"Suggested STEP   = {suggested_step} (Baseline: {BASELINE_STEP})")
    
    # --- VISUAL VALIDATION ---
    vis = img_bgr.copy()
    for p in best_coords:
        cv2.circle(vis, p, 5, (0, 255, 0), -1)
        cv2.rectangle(vis, (p[0]-24, p[1]-24), (p[0]+24, p[1]+24), (255, 0, 255), 1)
    
    out_path = os.path.join(cfg.DATA_DIRS["CALIB"], f"run_{run_id}_grid_discovery.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\nVisual check saved to Calibration Vault: {os.path.basename(out_path)}")

    # --- INTERPRETATION & NEXT STEPS ---
    print("\n" + "="*50)
    print("--- INTERPRETATION & NEXT STEPS ---")
    
    # Allow 2 pixels of leniency for X/Y and 0.5 for the Step
    if abs(suggested_x - BASELINE_X) <= 2 and abs(suggested_y - BASELINE_Y) <= 2 and abs(suggested_step - BASELINE_STEP) <= 0.5:
        print("[OK] Grid alignment matches the established baseline.")
        print("Action: No code updates required. You may proceed directly to Step 1.")
    else:
        print("[!] ALERT: Significant grid shift detected!")
        print("Action 1: Review the visual check image in Data_04_Calibration_Vault to ensure")
        print("          the purple boxes are perfectly centered on the ores.")
        print("Action 2: If the boxes are centered, you MUST update the ORE0_X, ORE0_Y,")
        print("          and STEP constants in your pipeline scripts (Steps 1 through 6)")
        print("          to match the 'Suggested' values above before proceeding to Step 1.")
    print("="*50)

if __name__ == "__main__":
    run_auto_discovery()