# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy and match confidence.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
ORE_TPL_DIR = os.path.join(cfg.TEMPLATE_DIR, "ores")
AI_DIM = 48
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_ore_templates():
    tpls = {}
    if not os.path.exists(ORE_TPL_DIR):
        return tpls
    for f in os.listdir(ORE_TPL_DIR):
        if f.endswith('.png'):
            name = f.replace('.png', '')
            img = cv2.imread(os.path.join(ORE_TPL_DIR, f), 0)
            if img is not None:
                tpls[name] = cv2.resize(img, (AI_DIM, AI_DIM))
    return tpls

def run_ore_audit():
    if not os.path.exists(STEP1_CSV):
        print("Error: Step 1 CSV missing.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    df = pd.read_csv(STEP1_CSV).sample(min(500, len(pd.read_csv(STEP1_CSV))))
    templates = load_ore_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID ACCURACY AUDIT ({len(df)} frames) ---")
    
    results = []
    row4_y = int(ORE0_Y + (3 * STEP))

    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, row['filename']), 0)
        if img is None: continue
        
        for col in range(6):
            x = int(ORE0_X + (col * STEP))
            x1, y1 = int(x - AI_DIM//2), int(row4_y - AI_DIM//2)
            roi = img[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
            
            if roi.shape != (AI_DIM, AI_DIM): continue

            # Compare against all templates
            matches = []
            for name, tpl in templates.items():
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCORR_NORMED, mask=mask)
                _, val, _, _ = cv2.minMaxLoc(res)
                matches.append((name, val))
            
            # Sort by confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            top_name, top_val = matches[0] if matches else ("unknown", 0.0)
            second_name, second_val = matches[1] if len(matches) > 1 else ("none", 0.0)
            
            results.append({
                'frame': row['frame_idx'],
                'slot': col,
                'top_ore': top_name,
                'top_conf': top_val,
                'second_ore': second_name,
                'second_conf': second_val,
                'margin': top_val - second_val
            })

    audit_df = pd.DataFrame(results)
    
    # Report Findings
    print("\n--- CONFIDENCE STATS BY ORE TYPE ---")
    stats = audit_df.groupby('top_ore')['top_conf'].agg(['mean', 'min', 'count'])
    print(stats)
    
    print("\n--- AMBIGUITY WARNINGS (Margin < 0.10) ---")
    ambiguous = audit_df[audit_df['margin'] < 0.10]
    print(f"Total Ambiguous Matches: {len(ambiguous)}")
    if not ambiguous.empty:
        print(ambiguous[['top_ore', 'second_ore', 'margin']].value_counts().head(5))

    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_accuracy_report.csv"), index=False)
    print(f"\n[DONE] Audit report saved to {OUT_DIR}/ore_id_accuracy_report.csv")

if __name__ == "__main__":
    run_ore_audit()