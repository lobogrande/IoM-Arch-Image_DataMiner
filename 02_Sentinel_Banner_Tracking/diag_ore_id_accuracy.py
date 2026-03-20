# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using DNA-Gated Masked Competitive Logic.
# Version: 3.9 (Parallelized Execution & Optimized Template Handling)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
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

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir):
    """Worker function to process 6 slots for a single frame."""
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    
    img_gray = cv2.imread(img_path, 0)
    if img_gray is None: return []
    
    # Get DNA for this specific frame
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        x1, y1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
        roi = img_gray[y1 : y1 + AI_DIM, x1 : x1 + AI_DIM]
        
        # DNA GATE
        if r4_dna[col] == '0':
            frame_results.append({
                'frame': f_idx, 'slot': col, 'detected': 'empty_bg',
                'ore_score': 0.0, 'bg_score': 1.0, 'margin': -1.0, 'source': 'dna_gate'
            })
            continue

        # BACKGROUND DEFENSE
        best_bg = {'id': 'none', 'score': 0.0}
        for bg_tpl in templates['bg']:
            res = cv2.matchTemplate(roi, bg_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_bg['score']: best_bg = {'id': bg_tpl['id'], 'score': val}

        # ORE IDENTIFICATION
        best_ore = {'tier': 'empty', 'score': 0.0, 'id': 'none', 'raw_score': 0.0}
        for tier, states in templates['ore'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(roi, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    raw = score
                    # Apply Shadow Compensation (Tuned to 1.03 for stability)
                    if state == 'sha': score *= 1.03
                    
                    if score > best_ore['score']:
                        best_ore = {'tier': tier, 'score': score, 'id': ore_tpl['id'], 'raw_score': raw}

        # COMPETITIVE LOGIC
        is_valid = (best_ore['score'] > 0.82) and (best_ore['score'] - best_bg['score'] > 0.08)
        detected = best_ore['tier'] if is_valid else "low_conf_ore"

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'ore_score': round(best_ore['score'], 4),
            'bg_score': round(best_bg['score'], 4),
            'margin': round(best_ore['score'] - best_bg['score'], 4),
            'ore_id': best_ore['id'], 'bg_id': best_bg['id'], 'source': 'masked_id'
        })

    return frame_results

def run_ore_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV):
        print("Error: Required CSV files missing.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    dna_df = pd.read_csv(DNA_CSV)
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    
    df = pd.read_csv(STEP1_CSV)
    # Increase sample size now that we are parallelized
    df_sample = df.sample(min(1000, len(df)))
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v3.9: PARALLELIZED GATED LOGIC ---")
    print(f"Analyzing {len(df_sample)} frames using multi-core processing...")

    all_results = []
    
    # Use ProcessPoolExecutor for CPU-bound template matching
    # We pass the shared data to a partial function to simplify the map call
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Convert df rows to list of dicts for serialization
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v3.9_parallel.csv"), index=False)
    
    print("\n--- DETECTION SUMMARY (v3.9) ---")
    print(audit_df['detected'].value_counts())
    
    print("\n--- TOP FALSE POSITIVE SUSPECTS (Margins) ---")
    div_hits = audit_df[audit_df['detected'].str.startswith('div', na=False)]
    if not div_hits.empty:
        print(div_hits.groupby('detected')['margin'].describe())

    print(f"\n[DONE] Scan complete. Output: {OUT_DIR}/ore_id_v3.9_parallel.csv")

if __name__ == "__main__":
    # Multiprocessing requires the main guard on Windows/macOS
    run_ore_audit()