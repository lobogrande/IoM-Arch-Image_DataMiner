# diag_ore_forensics.py
# Purpose: Deep mathematical analysis of specific missed identification slots.
# Version: 1.0 (Data-Driven Error Discovery)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- TARGETS FOR ANALYSIS ---
# Format: (Floor_ID, Row_Index, Col_Index, Expected_Tier)
TARGET_SLOTS = [
    (24, 2, 5, "com2"),  # F24 R3_S5
    (24, 3, 2, "dirt3"), # F24 R4_S2
    (27, 1, 3, "com2"),  # F27 R2_S3
    (37, 3, 3, "dirt3"),  # F37 R4_S3 (Common pattern)
    (45, 3, 2, "dirt3"),  # F45 R4_S2 (Common pattern)
    (52, 3, 3, "dirt3"),  # F52 R4_S3 (Common pattern)
    (54, 3, 2, "dirt3"),  # F54 R4_S2 (Common pattern)
    (59, 3, 2, "dirt3"),  # F59 R4_S2 (Common pattern)
    (69, 3, 2, "dirt3"),  # F69 R4_S2 (Common pattern)
    (70, 3, 2, "dirt3"),  # F70 R4_S2 (Common pattern)
    (83, 3, 3, "dirt3"),  # F83 R4_S3 (Common pattern)
    (88, 3, 3, "dirt3"),  # F88 R4_S3 (Common pattern)
    (91, 3, 2, "dirt3")  # F91 R4_S2 (Common pattern)
]

ORE0_X, ORE0_Y = 74, 261
STEP = 59.0
SIDE_PX = 48
MAX_SAMPLES = 40

def get_spatial_mask(r_idx):
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if r_idx == 0: mask[0:20, :] = 0
    return mask

def run_forensics():
    # Load all templates for comparison
    res = {'active': {}}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if "_act_plain_" in f and not any(x in f for x in ["bg", "player", "neg"]):
            img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
            tier = f.split("_")[0]
            if tier not in res['active']: res['active'][tier] = []
            res['active'][tier].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))

    boundaries = pd.read_csv(os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv"))
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])

    print(f"--- STARTING ORE FORENSIC ANALYSIS ---")
    
    for f_id, r_idx, col_idx, expected in TARGET_SLOTS:
        row = boundaries[boundaries['floor_id'] == f_id].iloc[0]
        start_f = int(row['true_start_frame'])
        mask = get_spatial_mask(r_idx)
        
        cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
        y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2)

        print(f"\n[ANALYZING Floor {f_id} | R{r_idx+1}_S{col_idx} | Expected: {expected}]")
        
        frame_logs = []
        for i in range(start_f, start_f + MAX_SAMPLES):
            img = cv2.imread(os.path.join(buffer_dir, all_files[i]), 0)
            if img is None: continue
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            
            # Metrics
            comp = cv2.Laplacian(roi, cv2.CV_64F).var()
            bright = cv2.mean(roi, mask=mask)[0]
            
            # Leaderboard
            candidates = []
            for tier, tpls in res['active'].items():
                score = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED, mask=mask))[1] for t in tpls])
                candidates.append({'tier': tier, 'score': score})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top = candidates[0]
            
            frame_logs.append({
                'frame': i, 'top_tier': top['tier'], 'top_score': round(top['score'], 4),
                'expected_score': round(next(x['score'] for x in candidates if x['tier'] == expected), 4),
                'complexity': int(comp), 'brightness': int(bright)
            })

        df = pd.DataFrame(frame_logs)
        print(f"  Complexity Window: {df['complexity'].min()} to {df['complexity'].max()}")
        print(f"  Brightness Window: {df['brightness'].min()} to {df['brightness'].max()}")
        print(f"  Average '{expected}' Score: {df['expected_score'].mean():.4f}")
        print(f"  Top Competitor: {df['top_tier'].mode()[0]} (Avg: {df['top_score'].mean():.4f})")
        
        if df['expected_score'].mean() < 0.42:
            print("  [!] CONCLUSION: Signal too weak. Check if an icon is covering the core.")

if __name__ == "__main__":
    run_forensics()