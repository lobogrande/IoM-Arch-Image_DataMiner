# diag_block_id_accuracy.py
# Purpose: Forensic audit of Row 4 block identification accuracy using DNA-Gated Masked Competitive Logic.
# Version: 3.8 (DNA Bit-Masking & Precision Resolution)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "block_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "debug_visuals")
AI_DIM = 48
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'block': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        if img.shape != (AI_DIM, AI_DIM): img = cv2.resize(img, (AI_DIM, AI_DIM))
            
        if f.startswith("background") or f.startswith("negative_ui"):
            templates['bg'].append({'id': f, 'img': img})
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['block']: templates['block'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['block'][tier][state].append({'id': f, 'img': img})
    return templates

def run_block_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV):
        print("Error: Required CSV files missing (Step 1 or Step 2 output).")
        return

    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    # 1. Load Step 2 DNA Signatures
    dna_df = pd.read_csv(DNA_CSV)
    # Map frame_idx -> r4_dna string (e.g., '001111')
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    
    # 2. Sample Step 1 frames
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(200, len(df)))
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    results = []
    row4_y = int(ORE0_Y + (3 * STEP))

    print(f"--- ORE ID AUDIT v3.8: DNA-GATED COMPETITIVE LOGIC ---")
    print(f"Active Tiers: {len(templates['block'])} | Background Templates: {len(templates['bg'])}")

    for _, row in df_sample.iterrows():
        f_idx = row['frame_idx']
        img_color = cv2.imread(os.path.join(buffer_dir, row['filename']))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
        if img_gray is None: continue
        
        # Get DNA for this specific frame
        r4_dna = dna_map.get(f_idx, "000000")

        for col in range(6):
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
            roi = img_gray[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
            
            # --- GATE: Is the slot even occupied? ---
            if r4_dna[col] == '0':
                results.append({
                    'frame': f_idx, 'slot': col, 'detected': 'empty_bg',
                    'ore_score': 0.0, 'bg_score': 1.0, 'margin': -1.0, 'source': 'dna_gate'
                })
                continue

            # --- IDENTIFICATION: Only run if DNA says '1' ---
            # 1. Best Background Defense
            best_bg = {'id': 'none', 'score': 0.0}
            for bg_tpl in templates['bg']:
                res = cv2.matchTemplate(roi, bg_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > best_bg['score']: best_bg = {'id': bg_tpl['id'], 'score': val}

            # 2. Best Block Identity
            best_block = {'tier': 'empty', 'score': 0.0, 'id': 'none'}
            for tier, states in templates['block'].items():
                for state in ['act', 'sha']:
                    for block_tpl in states[state]:
                        res = cv2.matchTemplate(roi, block_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                        _, score, _, _ = cv2.minMaxLoc(res)
                        if state == 'sha': score *= 1.05
                        if score > best_block['score']:
                            best_block = {'tier': tier, 'score': score, 'id': block_tpl['id']}

            # 3. Competitive Logic
            # Higher strictness since we already know the slot is occupied
            is_valid = (best_block['score'] > 0.82) and (best_block['score'] - best_bg['score'] > 0.08)
            detected = best_block['tier'] if is_valid else "low_conf_block"
            
            color = (0, 255, 0) if is_valid else (0, 165, 255) # Green for ID, Orange for low-conf
            cv2.rectangle(img_color, (x1, y1), (x1+AI_DIM, y1+AI_DIM), color, 1)
            cv2.putText(img_color, f"{detected}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            results.append({
                'frame': f_idx, 'slot': col, 'detected': detected,
                'ore_score': round(best_block['score'], 4),
                'bg_score': round(best_bg['score'], 4),
                'margin': round(best_block['score'] - best_bg['score'], 4),
                'block_id': best_block['id'], 'bg_id': best_bg['id'], 'source': 'masked_id'
            })

        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"audit_v38_f{f_idx}.jpg"), img_color)

    audit_df = pd.DataFrame(results)
    audit_df.to_csv(os.path.join(OUT_DIR, "block_id_v3.8_gated.csv"), index=False)
    
    print("\n--- GATED DETECTION SUMMARY (v3.8) ---")
    print(audit_df['detected'].value_counts())
    print(f"\n[DONE] Check 'block_id_v3.8_gated.csv'. 'low_conf_block' counts should be investigated.")

if __name__ == "__main__":
    run_block_audit()