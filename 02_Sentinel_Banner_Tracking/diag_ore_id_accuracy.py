# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 7.9 (Optical Peak Search: Finding the Scale & Y-Offset Sweet Spot)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "identity_verification")

# --- GROUND TRUTH SECTION ---
GROUND_TRUTH = {
    (0, 0): 'empty_dna', (0, 1): 'empty_dna', (0, 2): 'dirt1', (0, 3): 'com1', (0, 4): 'com1', (0, 5): 'dirt1',
    (1, 0): 'empty_dna', (1, 1): 'empty_dna', (1, 2): 'dirt1', (1, 3): 'com1', (1, 4): 'com1', (1, 5): 'dirt1',
    (2, 0): 'empty_dna', (2, 1): 'empty_dna', (2, 2): 'dirt1', (2, 3): 'com1', (2, 4): 'com1', (2, 5): 'dirt1',
    (121, 0): 'dirt1', (121, 1): 'dirt1', (121, 2): 'empty_dna', (121, 3): 'empty_dna', (121, 4): 'empty_dna', (121, 5): 'dirt1',
    (264, 0): 'empty_dna', (264, 1): 'dirt2', (264, 2): 'empty_dna', (264, 3): 'epic1', (264, 4): 'dirt2', (264, 5): 'empty_dna'
}

# ROI CONSTANTS
DIM_ID  = 48  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
X_JITTER = 2 
Y_JITTER = 2 # New vertical sweep to find vertical alignment peak

# SCALING CONSTANTS - Shifted up based on v7.8 results
SCALES = [1.05, 1.10, 1.15, 1.20, 1.25]

def get_spatial_mask(dim):
    mask = np.zeros((dim, dim), dtype=np.uint8)
    radius = int(18 * (dim / 48))
    cv2.circle(mask, (dim//2, dim//2), radius, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore_id': {}}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        if f.startswith(("background", "negative_ui")): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        parts = f.split("_")
        if len(parts) < 2: continue
        tier = parts[0]
        if tier not in templates['ore_id']:
            templates['ore_id'][tier] = []
            
        for s in SCALES:
            new_dim = int(DIM_ID * s)
            img_scaled = cv2.resize(img_raw, (new_dim, new_dim), interpolation=cv2.INTER_AREA)
            mask_scaled = get_spatial_mask(new_dim)
            templates['ore_id'][tier].append({
                'id': f, 'img': img_scaled, 'mask': mask_scaled, 'scale': s, 'tier': tier
            })
    return templates

def process_single_frame(frame_data, dna_map, templates, buffer_dir):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y_base = int(ORE0_Y + (3 * STEP))
    slot_matches = {}

    for col in range(6):
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna', 'candidates': []}
            continue

        cx = int(ORE0_X + (col * STEP))
        all_candidates = []
        
        for tier, variants in templates['ore_id'].items():
            for tpl in variants:
                side = tpl['img'].shape[0]
                # ROI covers X and Y jitter
                pad = 4 
                x1, y1 = int(cx - (side//2) - X_JITTER), int(row4_y_base - (side//2) - Y_JITTER)
                roi = img_gray[y1 : y1 + side + (Y_JITTER*2), x1 : x1 + side + (X_JITTER*2)]
                
                if roi.shape[0] < side or roi.shape[1] < side: continue
                
                res = cv2.matchTemplate(roi, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=tpl['mask'])
                _, score, _, max_loc = cv2.minMaxLoc(res)
                
                # max_loc tells us the local jitter offset (0,0 is top-left of search window)
                # x_off = max_loc[0] - X_JITTER, y_off = max_loc[1] - Y_JITTER
                y_offset = max_loc[1] - Y_JITTER

                all_candidates.append({
                    'tier': tier, 'id': tpl['id'], 'score': score, 
                    'scale': tpl['scale'], 'y_off': y_offset
                })
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        slot_matches[col] = {'status': 'occupied', 'candidates': all_candidates}

    frame_results = []
    has_detections = False
    
    for col in range(6):
        data = slot_matches.get(col)
        if not data: continue
        
        is_valid = False
        if data['status'] == 'empty_dna':
            detected = 'empty_dna'
            final = {'tier': 'empty_dna', 'score': 0.0, 'scale': 1.0, 'y_off': 0}
        else:
            final = data['candidates'][0]
            is_valid = final['score'] > 0.78 # Slightly lowered gate to see peaks better
            detected = final['tier'] if is_valid else "low_conf_id"
        
        # --- GROUND TRUTH FORENSICS ---
        truth_tier = GROUND_TRUTH.get((f_idx, col))
        truth_data = {'rank': -1, 'score': 0.0, 'scale': 0.0, 'y_off': 0}
        if truth_tier and truth_tier != 'empty_dna':
            for rank, c in enumerate(data.get('candidates', [])):
                if c['tier'] == truth_tier:
                    truth_data = {'rank': rank + 1, 'score': round(c['score'], 4), 'scale': c['scale'], 'y_off': c['y_off']}
                    break
        
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        if detected == 'empty_dna': color = (100, 100, 100)
        
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y_base - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        
        label = f"{detected} ({final['score']:.2f})"
        if truth_tier and detected != truth_tier:
            cv2.rectangle(img_color, (rx1-2, ry1-2), (rx1+DIM_ID+2, ry1+DIM_ID+2), (255, 0, 0), 1)
            label += f" [T:{truth_data['rank']} @{truth_data['scale']} Y:{truth_data['y_off']}]"
        
        cv2.putText(img_color, label, (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        if is_valid: has_detections = True

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected, 
            'score': round(final['score'], 4), 'win_scale': final['scale'],
            'truth_tier': truth_tier if truth_tier else 'none',
            'truth_rank': truth_data['rank'], 'truth_score': truth_data['score'],
            'truth_scale': truth_data['scale'], 'truth_y_off': truth_data['y_off']
        })

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"peak_v79_f{f_idx}.jpg"), img_color)
    return frame_results

def run_surgical_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    truth_frames = list(set([k[0] for k in GROUND_TRUTH.keys()]))
    df_sample = df[df['frame_idx'].isin(truth_frames)]
    if len(df_sample) < 100:
        remaining = df[~df['frame_idx'].isin(truth_frames)].sample(min(100 - len(df_sample), len(df)))
        df_sample = pd.concat([df_sample, remaining])

    print(f"--- ORE ID AUDIT v7.9: OPTICAL PEAK SEARCH ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v7.9_forensic.csv"), index=False)
    
    print(f"\n--- PEAK ERROR ANALYSIS ---")
    gt_only = audit_df[audit_df['truth_tier'] != 'none']
    if not gt_only.empty:
        ores_only = gt_only[gt_only['truth_tier'] != 'empty_dna']
        if not ores_only.empty:
            print(f"Avg Truth Rank: {ores_only['truth_rank'].mean():.1f}")
            print("\nTruth Peak Scales:")
            print(ores_only['truth_scale'].value_counts())
            print("\nTruth Peak Y-Offsets:")
            print(ores_only['truth_y_off'].value_counts())
    print(f"\n--- DETECTION SUMMARY ---")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_surgical_audit()