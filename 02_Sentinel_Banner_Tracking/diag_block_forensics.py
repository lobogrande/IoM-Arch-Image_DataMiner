# diag_block_forensics.py
# Purpose: Extract hard mathematical profiles from problematic block slots.
# Focus: Crosshair Color Density, Dirt3 Complexity, and Mod Zone Interference.
# Version: 1.2 (Target Correction & Saturation Profiling)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# Corrected Target Frames based on manual verification.
# Slots are Row 4 local indices (0-5).
TARGET_FRAMES = {
    196:  [1, 3],     # Dirt2 comparison (stamina vs loot/speed mods)
    847:  [2, 4],     # Leg1 called as Dirt3
    912:  [5],        # Myth1 called as Leg1
    557:  [4],        # Real Red Crosshair
    10:   [2, 5],     # Ghost Crosshairs (Dirt1 missed)
    1895: [5],        # Correct Gold Crosshair (previously mis-identified as blue)
    1255: [2, 3],     # Real Gold Crosshairs
    1260: [2, 3]      # Real Gold Crosshairs (Rotation poisoning)
}

# Physical Constants
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
TARGET_SCALE = 1.20
DIM = int(48 * TARGET_SCALE)
ROW4_Y = int(ORE0_Y + (3 * STEP)) + 2

# Output Configuration
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "block_id_audit")

def get_complexity(img):
    """Calculates Laplacian variance as a measure of structural texture."""
    return cv2.Laplacian(img, cv2.CV_64F).var()

def analyze_color_vibrancy(roi_bgr):
    """
    Analyzes HSV space to distinguish between 'Vibrant UI' and 'Muted Blocks'.
    Returns pixel counts and mean saturation for each range.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Expanded ranges to catch pale yellow/gold and deep wrap-around reds
    ranges = {
        'GOLD': ([10, 80, 80], [45, 255, 255]), # Wider hue, lower sat floor
        'BLUE': ([90, 80, 80], [145, 255, 255]),
        'RED':  ([0, 80, 80], [10, 255, 255]),
        'RED_EXT': ([160, 80, 80], [180, 255, 255])
    }
    
    stats = {}
    for name, (low, high) in ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        px_count = cv2.countNonZero(mask)
        # Calculate mean saturation of only the pixels that matched the color
        mean_sat = cv2.mean(s, mask=mask)[0] if px_count > 0 else 0
        stats[name] = {'px': px_count, 'sat': round(mean_sat, 1)}
        
    return stats

def run_forensic_scan():
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(buffer_dir):
        print(f"Error: Buffer directory not found at {buffer_dir}")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    results = []
    print(f"--- STARTING ORE FORENSIC SCAN v1.2 ---")
    print(f"Analyzing {len(TARGET_FRAMES)} frames for mathematical anomalies...\n")

    for f_idx, slots in TARGET_FRAMES.items():
        if f_idx >= len(all_files): continue
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for col in slots:
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - DIM//2), int(ROW4_Y - DIM//2)
            
            if y1 < 0 or y1+DIM > img.shape[0] or x1 < 0 or x1+DIM > img.shape[1]: continue
                
            roi_bgr = img[y1:y1+DIM, x1:x1+DIM]
            roi_gray = gray[y1:y1+DIM, x1:x1+DIM]

            # 1. Structural Analysis
            complexity = get_complexity(roi_gray)
            top_half, bot_half = roi_gray[0:DIM//2, :], roi_gray[DIM//2:DIM, :]
            top_energy, bot_energy = get_complexity(top_half), get_complexity(bot_half)
            
            # 2. Vibrancy & Color Profiling
            c_stats = analyze_color_vibrancy(roi_bgr)
            
            results.append({
                'frame': int(f_idx), 'slot': int(col),
                'complexity': round(complexity, 2),
                'energy_ratio': round(top_energy/max(1, bot_energy), 2),
                'gold_px': c_stats['GOLD']['px'], 'gold_sat': c_stats['GOLD']['sat'],
                'blue_px': c_stats['BLUE']['px'], 'blue_sat': c_stats['BLUE']['sat'],
                'red_px': c_stats['RED']['px'] + c_stats['RED_EXT']['px'],
                'red_sat': max(c_stats['RED']['sat'], c_stats['RED_EXT']['sat'])
            })

    df = pd.DataFrame(results)
    
    print("--- CROSSHAIR VIBRANCY PROFILE ---")
    xh_frames = [557, 1895, 1255, 10]
    for _, r in df[df['frame'].isin(xh_frames)].iterrows():
        status = "REAL" if int(r['frame']) != 10 else "GHOST"
        print(f"F{int(r['frame'])} S{int(r['slot'])} [{status}]: Gold(Px:{r['gold_px']}, Sat:{r['gold_sat']}) Blue(Px:{r['blue_px']}, Sat:{r['blue_sat']}) Red(Px:{r['red_px']}, Sat:{r['red_sat']})")

    print("\n--- MOD ZONE INTERFERENCE PROFILE ---")
    for _, r in df[df['frame'] == 196].iterrows():
        mod = "Stamina" if int(r['slot']) == 3 else "Loot/Speed"
        print(f"F196 S{int(r['slot'])} [{mod}]: Complexity:{r['complexity']} Energy Ratio:{r['energy_ratio']}")

    print("\n--- TIER COMPLEXITY CEILING ---")
    for _, r in df[df['frame'].isin([847, 912])].iterrows():
        print(f"F{int(r['frame'])} S{int(r['slot'])}: Complexity: {r['complexity']}")

    report_path = os.path.join(OUT_DIR, "forensic_anomaly_report_v1.2.csv")
    df.to_csv(report_path, index=False)
    print(f"\nForensic report saved to: {report_path}")

if __name__ == "__main__":
    run_forensic_scan()