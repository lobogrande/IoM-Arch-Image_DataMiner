# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Unified Competitive Profiling.
# Version: 4.1 (Self-Gated Parallelized Logic)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "debug_visuals")
AI_DIM = 48
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# THRESHOLDS
BG_OCCUPANCY_FLOOR = 0.92  # If BG match > this, slot is definitely empty
ORE_STRICT_GATE = 0.82     # Ore match must beat this
MARGIN_REQUIREMENT = 0.08  # Ore must beat BG by this much

def get_spatial_mask():
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore': {}, 'bg': []}
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
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['ore'][tier][state].append({'id': f, 'img': img})
    return templates

def process_single_frame(frame_data, templates, mask, buffer_dir):
    """Worker function to process 6 slots using self-gating logic."""
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    
    img_gray = cv2.imread(img_path, 0)
    if img_gray is None: return []
    
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        x1, y1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
        roi = img_gray[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
        
        # 1. Background Defense (The new Gate)
        best_bg = {'id': 'none', 'score': 0.0}
        for bg_tpl in templates['bg']:
            res = cv2.matchTemplate(roi, bg_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_bg['score']: best_bg = {'id': bg_tpl['id'], 'score': val}

        # SELF-GATE: If background match is pristine, skip ores
        if best_bg['score'] > BG_OCCUPANCY_FLOOR:
            frame_results.append({
                'frame': f_idx, 'slot': col, 'detected': 'empty_bg',
                'ore_score': 0.0, 'bg_score': round(best_bg['score'], 4), 
                'margin': -1.0, 'ore_id': 'none', 'bg_id': best_bg['id']
            })
            continue

        # 2. Ore Identification
        best_ore = {'tier': 'empty', 'score': 0.0, 'id': 'none'}
        for tier, states in templates['ore'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(roi, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if state == 'sha': score *= 1.03
                    
                    if score > best_ore['score']:
                        best_ore = {'tier': tier, 'score': score, 'id': ore_tpl['id']}

        # 3. Competitive Logic
        margin = best_ore['score'] - best_bg['score']
        is_valid = (best_ore['score'] > ORE_STRICT_GATE) and (margin > MARGIN_REQUIREMENT)
        detected = best_ore['tier'] if is_valid else "low_conf_ore"

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'ore_score': round(best_ore['score'], 4),
            'bg_score': round(best_bg['score'], 4),
            'margin': round(margin, 4),
            'ore_id': best_ore['id'], 'bg_id': best_bg['id']
        })

    return frame_results

def run_ore_audit():
    if not os.path.exists(STEP1_CSV):
        print("Error: Step 1 CSV missing.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v4.1: UNIFIED COMPETITIVE LOGIC ---")
    print(f"Analyzing {len(df_sample)} frames...")

    all_results = []
    worker_func = partial(process_single_frame, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"Worker Exception: {e}")
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v4.1_unified.csv"), index=False)
    
    print("\n--- UNIFIED DETECTION SUMMARY (v4.1) ---")
    print(audit_df['detected'].value_counts())
    
    if 'low_conf_ore' in audit_df['detected'].values:
        print("\n--- LOW CONFIDENCE ANALYSIS ---")
        low_conf = audit_df[audit_df['detected'] == 'low_conf_ore']
        print(f"Average Margin for Low-Conf Ores: {low_conf['margin'].mean():.4f}")
        print(f"Max Margin for Low-Conf Ores:     {low_conf['margin'].max():.4f}")

    print(f"\n[DONE] Audit complete. Output: {OUT_DIR}/ore_id_v4.1_unified.csv")

if __name__ == "__main__":
    run_ore_audit()