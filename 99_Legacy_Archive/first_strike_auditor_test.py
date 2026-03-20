# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Masked Competitive Logic.
# Version: 3.6 (Visual Debug & Raw Delta Profiling)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "debug_visuals")
AI_DIM = 48
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        if img.shape != (AI_DIM, AI_DIM): img = cv2.resize(img, (AI_DIM, AI_DIM))
        if f.startswith("background"):
            templates['bg'].append(img)
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['ore'][tier][state].append(img)
    return templates

def run_ore_audit():
    if not os.path.exists(STEP1_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    df = pd.read_csv(STEP1_CSV)
    # We take a specific sample including some suspected boss floors if possible
    df_sample = df.sample(min(200, len(df)))
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    results = []
    row4_y = int(ORE0_Y + (3 * STEP))

    print(f"--- ORE ID AUDIT v3.6: VISUAL DEBUG MODE ---")

    for idx, row in df_sample.iterrows():
        img_color = cv2.imread(os.path.join(buffer_dir, row['filename']))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
        if img_gray is None: continue
        
        frame_results = []
        for col in range(6):
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
            roi = img_gray[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
            
            # 1. Background Match
            best_bg_score = 0.0
            for bg_img in templates['bg']:
                bg_res = cv2.matchTemplate(roi, bg_img, cv2.TM_CCOEFF_NORMED).max()
                if bg_res > best_bg_score: best_bg_score = bg_res

            # 2. Ore Match
            best_ore = {'tier': 'empty', 'score': 0.0, 'state': 'none', 'raw': 0.0}
            for tier, states in templates['ore'].items():
                for state in ['act', 'sha']:
                    for t_img in states[state]:
                        res = cv2.matchTemplate(roi, t_img, cv2.TM_CCORR_NORMED, mask=mask)
                        _, score, _, _ = cv2.minMaxLoc(res)
                        raw_score = score
                        if state == 'sha': score *= 1.05
                        
                        if score > best_ore['score']:
                            best_ore = {'tier': tier, 'score': score, 'state': state, 'raw': raw_score}

            # 3. Competitive Logic
            is_valid = (best_ore['score'] > 0.80) and (best_ore['score'] - best_bg_score > 0.06)
            detected = best_ore['tier'] if is_valid else "empty"
            
            # HUD Visual
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.rectangle(img_color, (x1, y1), (x1+AI_DIM, y1+AI_DIM), color, 1)
            cv2.putText(img_color, f"{detected}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(img_color, f"{best_ore['score']:.2f}", (x1, y1+AI_DIM+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            results.append({
                'frame': row['frame_idx'], 'slot': col, 'detected': detected,
                'score': best_ore['score'], 'bg': best_bg_score, 'margin': best_ore['score'] - best_bg_score
            })

        # Save debug image
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"audit_f{row['frame_idx']}.jpg"), img_color)

    audit_df = pd.DataFrame(results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v3.6_debug.csv"), index=False)
    print(f"\n[DONE] Saved {len(df_sample)} debug images to {DEBUG_IMG_DIR}")
    print(f"Check the 'margin' column in the CSV for slots labeled 'div' that should be 'empty'.")

if __name__ == "__main__":
    run_ore_audit()