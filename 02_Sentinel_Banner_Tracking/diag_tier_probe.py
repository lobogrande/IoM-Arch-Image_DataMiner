# diag_tier_probe.py
# Purpose: Find the exact Scale, Coordinates, and Filter needed for 0.90+ confidence.
# Targets: Floor 1, Slot 1 (Known Common Block)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- TARGET CONSTANTS (Current Best Guess) ---
TARGET_FLOOR = 1
TARGET_SLOT = 1 # R1_S1
ORE_TIER = "com1" 

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def run_probe():
    buffer_dir = cfg.get_buffer_path(0)
    boundaries = pd.read_csv(os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv"))
    
    # Get the very first frame of Floor 1
    row = boundaries[boundaries['floor_id'] == TARGET_FLOOR].iloc[0]
    f_idx = int(row['true_start_frame'])
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Load the template
    t_path = os.path.join(cfg.TEMPLATE_DIR, f"{ORE_TIER}_act_plain_0.png")
    tpl_raw = cv2.imread(t_path, 0)
    
    if tpl_raw is None:
        print(f"Error: Could not find template at {t_path}")
        return

    print(f"--- STARTING MATHEMATICAL PROBE [F:{TARGET_FLOOR} S:{TARGET_SLOT}] ---")
    
    results = []
    
    # Test different Pre-processing methods
    filters = {
        "RAW": lambda x: x,
        "CLAHE": lambda x: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(x),
        "NORM": lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    }

    # Test different Scales (from 1.0 to 1.3)
    for scale in np.linspace(1.0, 1.3, 15):
        side = int(48 * scale)
        tpl_resized = cv2.resize(tpl_raw, (side, side))
        
        # Test small coordinate Jitters (5x5 pixels)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cx = int(ORE0_X + (TARGET_SLOT * STEP)) + dx
                cy = int(ORE0_Y) + dy
                x1, y1 = cx - side//2, cy - side//2
                roi = img_gray[y1:y1+side, x1:x1+side]
                
                if roi.shape != tpl_resized.shape: continue

                for f_name, f_func in filters.items():
                    roi_proc = f_func(roi)
                    tpl_proc = f_func(tpl_resized)
                    
                    res = cv2.matchTemplate(roi_proc, tpl_proc, cv2.TM_CCOEFF_NORMED)
                    score = cv2.minMaxLoc(res)[1]
                    
                    results.append({
                        'scale': round(scale, 3),
                        'filter': f_name,
                        'dx': dx, 'dy': dy,
                        'score': score
                    })

    df = pd.DataFrame(results)
    best = df.sort_values('score', ascending=False).iloc[0]
    
    print(f"\n[WINNER FOUND]")
    print(f"Confidence: {best['score']:.4f}")
    print(f"Optimal Scale: {best['scale']}")
    print(f"Optimal Filter: {best['filter']}")
    print(f"Coord Shift Needed: DX={best['dx']}, DY={best['dy']}")
    
    if best['score'] < 0.80:
        print("\n[CRITICAL WARNING]: Even with optimization, confidence is low.")
        print("This suggests the template and the video frames are from different resolutions/sources.")
    else:
        print("\n[PROCEED]: Use these constants to rebuild the Master Script.")

if __name__ == "__main__":
    run_probe()