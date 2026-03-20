# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Masked Competitive Logic.
# Version: 3.5 (Aligned with Production Mining v3.4)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
AI_DIM = 48
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_spatial_mask():
    """Creates the 48x48 circular mask to ignore corner icons/noise."""
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    """
    Loads templates following the user's proven tier_state_... convention.
    Returns: {'ore': {tier: {state: [imgs]}}, 'bg': [imgs]}
    """
    templates = {'ore': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    if not os.path.exists(t_path):
        print(f"Error: Template directory not found at {t_path}")
        return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        if img.shape != (AI_DIM, AI_DIM): 
            img = cv2.resize(img, (AI_DIM, AI_DIM))
            
        if f.startswith("background"):
            templates['bg'].append(img)
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            
            if tier not in templates['ore']: 
                templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: 
                templates['ore'][tier][state].append(img)
                
    return templates

def run_ore_audit():
    if not os.path.exists(STEP1_CSV):
        print("Error: Step 1 CSV missing.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    # Process a significant sample to catch outliers
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID ACCURACY AUDIT (Masked Competitive Logic) ---")
    print(f"Analyzing {len(df_sample)} frames against {len(templates['ore'])} ore tiers...")
    
    results = []
    row4_y = int(ORE0_Y + (3 * STEP))

    for idx, row in df_sample.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, row['filename']), 0)
        if img is None: continue
        
        for col in range(6):
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
            roi = img[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
            
            if roi.shape != (AI_DIM, AI_DIM): continue

            # 1. Find Best Background Match
            best_bg_score = 0.0
            for bg_img in templates['bg']:
                bg_res = cv2.matchTemplate(roi, bg_img, cv2.TM_CCOEFF_NORMED).max()
                if bg_res > best_bg_score: best_bg_score = bg_res

            # 2. Find Best Ore Match (Masked CCORR)
            best_ore = {'tier': 'empty', 'score': 0.0, 'state': 'none'}
            
            for tier, states in templates['ore'].items():
                for state in ['act', 'sha']:
                    for t_img in states[state]:
                        res = cv2.matchTemplate(roi, t_img, cv2.TM_CCORR_NORMED, mask=mask)
                        _, score, _, _ = cv2.minMaxLoc(res)
                        
                        # Apply Shadow Compensation
                        if state == 'sha': score *= 1.05
                        
                        if score > best_ore['score']:
                            best_ore = {'tier': tier, 'score': score, 'state': state}

            # 3. Apply Competitive Gate (0.80 threshold + 0.06 background delta)
            is_valid = (best_ore['score'] > 0.80) and (best_ore['score'] - best_bg_score > 0.06)
            detected_tier = best_ore['tier'] if is_valid else "empty"

            results.append({
                'frame': row['frame_idx'],
                'slot': col,
                'detected': detected_tier,
                'ore_score': round(best_ore['score'], 4),
                'bg_score': round(best_bg_score, 4),
                'margin': round(best_ore['score'] - best_bg_score, 4),
                'state': best_ore['state']
            })

    audit_df = pd.DataFrame(results)
    
    # Statistics
    print("\n--- DETECTION SUMMARY ---")
    print(audit_df['detected'].value_counts())
    
    print("\n--- CONFIDENCE METRICS (Valid Hits) ---")
    valid_hits = audit_df[audit_df['detected'] != 'empty']
    if not valid_hits.empty:
        stats = valid_hits.groupby('detected')['ore_score'].agg(['mean', 'min', 'count'])
        print(stats)
    else:
        print("[!] Warning: Zero ores detected even with competitive logic. Check template paths.")

    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v3.5_audit.csv"), index=False)
    print(f"\n[DONE] Audit report saved to {OUT_DIR}/ore_id_v3.5_audit.csv")

if __name__ == "__main__":
    run_ore_audit()