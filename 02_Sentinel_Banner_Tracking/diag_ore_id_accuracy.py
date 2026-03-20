# diag_ore_id_accuracy.py
# Purpose: Forensic Discrimination Audit for Row 4 Ore Identification.
# Version: 6.2 (Greediness Index & Structural Entropy Profiling)

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

# ROI CONSTANTS
DIM_ID  = 48  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# GAME PHYSICS: ORE FLOOR RESTRICTIONS
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_complexity(img):
    """Measures visual detail density (Laplacian variance)."""
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_family(tier_name):
    return ''.join([i for i in tier_name if not i.isdigit()])

def get_spatial_mask():
    mask = np.zeros((DIM_ID, DIM_ID), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
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
        
        img_id = cv2.resize(img_raw, (DIM_ID, DIM_ID))
        complexity = get_complexity(img_id)
        
        parts = f.split("_")
        if len(parts) < 2: continue
        tier, state = parts[0], parts[1]
        
        if tier not in templates['ore_id']:
            templates['ore_id'][tier] = []
        templates['ore_id'][tier].append({'id': f, 'img': img_id, 'comp': complexity})
    return templates

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_gray = cv2.imread(img_path, 0)
    if img_gray is None: return [], []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    
    forensic_results = []
    greedy_accumulator = []

    for col in range(6):
        if r4_dna[col] == '0': continue 

        cx = int(ORE0_X + (col * STEP))
        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        
        if search_id.shape[0] < DIM_ID or search_id.shape[1] < DIM_ID: continue
        roi_complexity = get_complexity(search_id)

        # MATCH ALL TIERS (Unfiltered for Forensics)
        all_matches = []
        for tier, states in templates['ore_id'].items():
            for ore_tpl in states:
                res = cv2.matchTemplate(search_id, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                score = cv2.minMaxLoc(res)[1]
                all_matches.append({
                    'tier': tier, 'id': ore_tpl['id'], 'score': score, 'comp': ore_tpl['comp']
                })
        
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        top_5 = all_matches[:5]
        greedy_accumulator.extend([m['id'] for m in all_matches[:3]])

        # Forensic Record
        res_row = {
            'frame': f_idx, 'slot': col,
            'winner': top_5[0]['tier'], 'w_score': round(top_5[0]['score'], 4),
            'roi_comp': round(roi_complexity, 1), 'tpl_comp': round(top_5[0]['comp'], 1),
            'comp_diff': round(abs(roi_complexity - top_5[0]['comp']), 1),
            'cand2': top_5[1]['tier'], 's2': round(top_5[1]['score'], 4),
            'cand3': top_5[2]['tier'], 's3': round(top_5[2]['score'], 4),
            'margin_1_2': round(top_5[0]['score'] - top_5[1]['score'], 4),
            'win_id': top_5[0]['id']
        }
        forensic_results.append(res_row)

    return forensic_results, greedy_accumulator

def run_forensic_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV):
        print("Error: Required files missing.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    
    # Target 200 frames for a deep profile
    df_sample = df.sample(min(200, len(df)))
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID FORENSIC AUDIT v6.2 ---")
    print(f"Goal: Identify Structural Mismatches and Greedy Noise Magnets")

    all_results, all_greedy = [], []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res, greedy = future.result()
            all_results.extend(res)
            all_greedy.extend(greedy)

    # 1. GREEDINESS INDEX
    greedy_counts = Counter(all_greedy)
    print("\n[FORENSIC] Greediness Index (Occurrences in Top 3):")
    for tid, count in greedy_counts.most_common(12):
        print(f"  {count:4d} hits: {tid}")

    # 2. STRUCTURAL GAP ANALYSIS
    audit_df = pd.DataFrame(all_results)
    print("\n[FORENSIC] Structural Complexity Mismatches (ROI vs Winner):")
    # Large comp_diff suggests a smooth template matched a busy ROI (The Dirt Trap)
    mismatches = audit_df[audit_df['comp_diff'] > 1000]
    print(f"  High Complexity Mismatches: {len(mismatches)} / {len(audit_df)}")
    
    # 3. CONSTRAINT VIOLATION CHECK
    print("\n[FORENSIC] Physical Constraint Violations (Unfiltered Run):")
    violations = 0
    for frame, group in audit_df.groupby('frame'):
        detected_tiers = group['winner'].unique()
        # Check One-Tier-Per-Family
        families = [get_family(t) for t in detected_tiers]
        if len(families) != len(set(families)):
            violations += 1
            continue
        # Check Floor Overlap
        if len(detected_tiers) > 1:
            ranges = [ORE_RESTRICTIONS.get(t, (1, 999)) for t in detected_tiers]
            max_start = max(r[0] for r in ranges)
            min_stop = min(r[1] for r in ranges)
            if max_start > min_stop:
                violations += 1

    print(f"  Rows violating game physics: {violations} / {len(audit_df['frame'].unique())}")
    
    out_path = os.path.join(OUT_DIR, "ore_discrimination_audit_v6.2.csv")
    audit_df.to_csv(out_path, index=False)
    print(f"\n[DONE] Diagnostic Complete. Check {out_path} for raw conflict data.")

if __name__ == "__main__":
    run_forensic_audit()