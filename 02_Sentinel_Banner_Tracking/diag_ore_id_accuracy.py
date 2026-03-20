# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using DNA-Aligned ROI.
# Version: 4.4 (Jitter-Search Alignment & Threshold Calibration)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "debug_visuals")

# ROI CONSTANTS
AI_DIM = 30  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 # Search +/- 2 pixels to find perfect alignment

# THRESHOLDS - Calibrated for surgical CCOEFF matching
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE = 0.50     # Lowered to account for Row 4 noise, relying on Margin instead
MARGIN_REQUIREMENT = 0.10  # Increased margin to ensure clear separation from BG

def load_all_templates():
    """
    Loads templates using a CENTER CROP to ensure 1:1 pixel parity with the 30x30 ROI.
    """
    templates = {'ore': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        
        h, w = img.shape
        if h > AI_DIM or w > AI_DIM:
            cy, cx = h // 2, w // 2
            r = AI_DIM // 2
            img = img[cy-r : cy+r, cx-r : cx+r]
        elif h < AI_DIM or w < AI_DIM:
            img = cv2.resize(img, (AI_DIM, AI_DIM))
            
        if f.startswith("background") or f.startswith("negative_ui"):
            templates['bg'].append({'id': f, 'img': img})
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['ore'][tier][state].append({'id': f, 'img': img})
    return templates

def process_single_frame(frame_data, templates, buffer_dir):
    """Worker function using Jitter-Search to find best pixel-parity match."""
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    
    img_gray = cv2.imread(img_path, 0)
    if img_gray is None: return []
    
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        
        # 1. Background Defense with Jitter Search
        # We look in a slightly larger area to find the best background fit
        x1, y1 = int(cx - (AI_DIM//2) - JITTER), int(row4_y - (AI_DIM//2) - JITTER)
        search_area = img_gray[y1 : y1 + AI_DIM + (JITTER*2), x1 : x1 + AI_DIM + (JITTER*2)]
        
        if search_area.shape[0] < AI_DIM or search_area.shape[1] < AI_DIM: continue

        best_bg = {'id': 'none', 'score': 0.0}
        for bg_tpl in templates['bg']:
            res = cv2.matchTemplate(search_area, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_bg['score']: best_bg = {'id': bg_tpl['id'], 'score': val}

        # SELF-GATE
        if best_bg['score'] > BG_OCCUPANCY_FLOOR:
            frame_results.append({
                'frame': f_idx, 'slot': col, 'detected': 'empty_bg',
                'ore_score': 0.0, 'bg_score': round(best_bg['score'], 4), 
                'margin': -1.0, 'ore_id': 'none', 'bg_id': best_bg['id']
            })
            continue

        # 2. Ore Identification with Jitter Search
        best_ore = {'tier': 'empty', 'score': 0.0, 'id': 'none'}
        for tier, states in templates['ore'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_area, ore_tpl['img'], cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    
                    if state == 'sha': score *= 1.02
                    
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
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v4.4: JITTER-SEARCH ALIGNMENT ---")
    print(f"ROI: {AI_DIM}x{AI_DIM} (+/- {JITTER}px) | Method: TM_CCOEFF_NORMED")

    all_results = []
    worker_func = partial(process_single_frame, templates=templates, buffer_dir=buffer_dir)
    
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
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v4.4_jitter.csv"), index=False)
    
    print("\n--- DETECTION SUMMARY (v4.4) ---")
    print(audit_df['detected'].value_counts())
    
    if 'low_conf_ore' in audit_df['detected'].values:
        low_conf = audit_df[audit_df['detected'] == 'low_conf_ore']
        print(f"\nLow-Conf Avg Margin: {low_conf['margin'].mean():.4f}")
        print(f"Low-Conf Max Margin: {low_conf['margin'].max():.4f}")

    print(f"\n[DONE] Check {OUT_DIR}/ore_id_v4.4_jitter.csv")

if __name__ == "__main__":
    run_ore_audit()