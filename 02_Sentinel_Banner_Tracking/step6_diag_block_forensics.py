# step6_diag_block_forensics.py
# Purpose: Deep mathematical analysis of specific missed identification slots.
# Version: 2.0 (Architecture Aligned & Step 6 Mirrored)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"final_floor_boundaries_run_{RUN_ID}.csv")

# --- VALIDATED GRID CONSTANTS ---
ORE0_X, ORE0_Y = 74, 261
STEP = 59.0
SIDE_PX = 48
MAX_SAMPLES = 40
ROTATION_VARIANTS = [-3, 0, 3]

# --- TARGETS FOR ANALYSIS ---
TARGET_SLOTS =[
    (24, 2, 5, "com2"),   # F24 R3_S5
    (24, 3, 2, "dirt3"),  # F24 R4_S2
    (27, 1, 3, "com2"),   # F27 R2_S3
    (37, 3, 3, "dirt3"),  # F37 R4_S3 
    (45, 3, 2, "dirt3"),  # F45 R4_S2 
    (52, 3, 3, "dirt3"),  # F52 R4_S3 
    (54, 3, 2, "dirt3"),  # F54 R4_S2 
    (59, 3, 2, "dirt3"),  # F59 R4_S2 
    (69, 3, 2, "dirt3"),  # F69 R4_S2 
    (70, 3, 2, "dirt3"),  # F70 R4_S2 
    (83, 3, 3, "dirt3"),  # F83 R4_S3 
    (88, 3, 3, "dirt3"),  # F88 R4_S3 
    (91, 3, 2, "dirt3")   # F91 R4_S2 
]

def rotate_image(image, angle):
    if angle == 0: return image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def get_spatial_mask(r_idx):
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if r_idx == 0: mask[0:20, :] = 0
    return mask

def run_forensics():
    if not os.path.exists(BOUNDARIES_CSV):
        print(f"Error: Could not find boundaries file {BOUNDARIES_CSV}")
        return

    # Load templates exactly how Step 6 does it (with rotations)
    res = {'active': {}}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if "_act_plain_" in f and not any(x in f for x in["bg", "player", "neg"]):
            img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
            if img is None: continue
            tier = f.split("_")[0]
            if tier not in res['active']: res['active'][tier] =[]
            img_scaled = cv2.resize(img, (SIDE_PX, SIDE_PX))
            for angle in ROTATION_VARIANTS:
                res['active'][tier].append(rotate_image(img_scaled, angle))

    boundaries = pd.read_csv(BOUNDARIES_CSV)
    all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg'))])

    print(f"--- STARTING ORE FORENSIC ANALYSIS (Run {RUN_ID}) ---")
    
    for f_id, r_idx, col_idx, expected in TARGET_SLOTS:
        # Check if floor exists in data
        match = boundaries[boundaries['floor_id'] == f_id]
        if match.empty:
            print(f"\n[!] Skipping Floor {f_id}: Not found in boundaries CSV.")
            continue
            
        row = match.iloc[0]
        start_f = int(row['true_start_frame'])
        mask = get_spatial_mask(r_idx)
        
        cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
        y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2)

        print(f"\n[ANALYZING Floor {f_id} | R{r_idx+1}_S{col_idx} | Expected: {expected}]")
        
        frame_logs =[]
        for i in range(start_f, start_f + MAX_SAMPLES):
            if i >= len(all_files): break
            img = cv2.imread(os.path.join(SOURCE_DIR, all_files[i]), 0)
            if img is None: continue
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            
            # Metrics
            comp = cv2.Laplacian(roi, cv2.CV_64F).var()
            bright = cv2.mean(roi, mask=mask)[0]
            
            # Leaderboard
            candidates =[]
            for tier, tpls in res['active'].items():
                score = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED, mask=mask))[1] for t in tpls])
                candidates.append({'tier': tier, 'score': score})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top = candidates[0]
            
            try:
                exp_score = next(x['score'] for x in candidates if x['tier'] == expected)
            except StopIteration:
                exp_score = 0.0 # Safety if expected tier template is missing
            
            frame_logs.append({
                'frame': i, 'top_tier': top['tier'], 'top_score': round(top['score'], 4),
                'expected_score': round(exp_score, 4),
                'complexity': int(comp), 'brightness': int(bright)
            })

        if not frame_logs:
            continue

        df = pd.DataFrame(frame_logs)
        print(f"  Complexity Window: {df['complexity'].min()} to {df['complexity'].max()}")
        print(f"  Brightness Window: {df['brightness'].min()} to {df['brightness'].max()}")
        print(f"  Average '{expected}' Score: {df['expected_score'].mean():.4f}")
        
        # Get the most common top tier to see who is bullying
        top_bully = df['top_tier'].mode()[0]
        bully_avg = df[df['top_tier'] == top_bully]['top_score'].mean()
        
        print(f"  Top Competitor: {top_bully} (Avg: {bully_avg:.4f})")
        print(f"  Margin (Bully - Expected): {bully_avg - df['expected_score'].mean():.4f}")
        
        if df['expected_score'].mean() < 0.40:
            print("  [!] CONCLUSION: Signal too weak. Target is heavily obscured or misaligned.")

if __name__ == "__main__":
    run_forensics()