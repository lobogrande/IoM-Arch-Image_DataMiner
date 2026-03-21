# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 8.9 (Adaptive Context Gating & Forensic Restoration)

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
Y_JITTER = 1 

# OPTICAL CONSTANTS
TARGET_SCALE = 1.20
ROW4_Y_PERSPECTIVE_SHIFT = 2 

# ADAPTIVE GATING
ACTIVE_GATE = 0.65
SHADOW_GATE = 0.42  # Lower floor for dark slots
Z_TRUST_THRESHOLD = 2.2 
TIER_CONF_BUFFER = 0.08 
COMPLEXITY_PENALTY_COEFF = 0.00025 

# BULLY PENALTY MAP
BULLY_PENALTIES = {
    'leg1': 0.08, 'leg2': 0.05, 'leg3': 0.10,
    'myth1': 0.04, 'myth2': 0.07, 'myth3': 0.08,
    'div3': 0.12, 'epic3': 0.05,
    'com3': 0.06, 'rare3': 0.06
}

# GAME PHYSICS: ORE FLOOR RESTRICTIONS
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def apply_gamma_lift(img, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def get_complexity(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_spatial_mask(dim, is_core_only=False):
    mask = np.zeros((dim, dim), dtype=np.uint8)
    base_r = 12 if is_core_only else 18
    radius = int(base_r * (dim / 48))
    cv2.circle(mask, (dim//2, dim//2), radius, 255, -1)
    top_exclude_limit = int(dim * 0.40)
    mask[0:top_exclude_limit, :] = 0
    return mask

def detect_crosshair(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return "none"
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        'GOLD': ([18, 100, 100], [35, 255, 255]),
        'BLUE': ([100, 100, 100], [130, 255, 255]),
        'RED':  ([0, 100, 100], [10, 255, 255])
    }
    for name, (low, high) in ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        if np.sum(mask) > 100: return name
    return "none"

def get_family(tier_name):
    return ''.join([i for i in tier_name if not i.isdigit()])

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
        new_dim = int(DIM_ID * TARGET_SCALE)
        img_scaled = cv2.resize(img_raw, (new_dim, new_dim), interpolation=cv2.INTER_AREA)
        lifted_tpl = apply_gamma_lift(img_scaled, 0.6)
        comp = get_complexity(lifted_tpl)
        templates['ore_id'][tier].append({
            'id': f, 'img': img_scaled, 'comp': comp,
            'mask_std': get_spatial_mask(new_dim, False),
            'mask_core': get_spatial_mask(new_dim, True),
            'tier': tier
        })
    return templates

def draw_shadow_text(img, text, pos, font, scale, color, thick):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale, (0,0,0), thick+1)
    cv2.putText(img, text, pos, font, scale, color, thick)

def process_single_frame(frame_data, dna_map, templates, buffer_dir):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    if img_color is None: return []
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y_base = int(ORE0_Y + (3 * STEP)) + ROW4_Y_PERSPECTIVE_SHIFT
    
    slot_matches = {}
    for col in range(6):
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna', 'candidates': [], 'xhair': 'none', 'is_sha': False}
            continue
        cx = int(ORE0_X + (col * STEP))
        side_px = int(DIM_ID * TARGET_SCALE)
        tx1, ty1 = int(cx - side_px//2), int(row4_y_base - side_px//2)
        roi_bgr = img_color[ty1 : ty1 + side_px, tx1 : tx1 + side_px]
        roi_gray = img_gray[ty1 : ty1 + side_px, tx1 : tx1 + side_px]
        
        is_sha = False
        if roi_gray.size > 0:
            mean_lum = np.mean(roi_gray)
            is_sha = mean_lum < 85
            roi_lifted = apply_gamma_lift(roi_gray, 0.5)
            roi_complexity = get_complexity(roi_lifted)
        else:
            roi_complexity = 0

        xhair = detect_crosshair(roi_bgr)
        use_core_mask = (xhair != "none")
        all_candidates = []
        for tier, variants in templates['ore_id'].items():
            bully_penalty = BULLY_PENALTIES.get(tier, 0.0)
            for tpl in variants:
                active_mask = tpl['mask_core'] if use_core_mask else tpl['mask_std']
                side = tpl['img'].shape[0]
                x1, y1 = int(cx - (side//2) - X_JITTER), int(row4_y_base - (side//2) - Y_JITTER)
                search_area = img_gray[y1 : y1 + side + (Y_JITTER*2), x1 : x1 + side + (X_JITTER*2)]
                if search_area.shape[0] < side or search_area.shape[1] < side: continue
                search_lifted = apply_gamma_lift(search_area, 0.5)
                res = cv2.matchTemplate(search_lifted, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=active_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                comp_diff = max(0, tpl['comp'] - roi_complexity)
                weighted_score = score - bully_penalty - (comp_diff * COMPLEXITY_PENALTY_COEFF)
                all_candidates.append({'tier': tier, 'id': tpl['id'], 'score': weighted_score})
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        scores = [c['score'] for c in all_candidates]
        z_score = (all_candidates[0]['score'] - np.mean(scores)) / np.std(scores) if (len(scores) > 1 and np.std(scores) > 0) else 0
        slot_matches[col] = {'status': 'occupied', 'candidates': all_candidates, 'z_score': z_score, 'xhair': xhair, 'is_sha': is_sha}

    # Resolution
    anchor = {'tier': 'none', 'score': -1.0, 'z': 0.0, 'range': (1, 999), 'col': -1}
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna' or not data['candidates']: continue
        top = data['candidates'][0]
        if data['z_score'] > anchor['z'] and data['z_score'] > Z_TRUST_THRESHOLD:
            anchor = {'tier': top['tier'], 'score': top['score'], 'z': data['z_score'],
                      'range': ORE_RESTRICTIONS.get(top['tier'], (1, 999)), 'col': col}

    frame_results = []
    has_detections = False
    family_champions = {}
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna' or not data['candidates']: continue
        for cand in data['candidates']:
            t_range = ORE_RESTRICTIONS.get(cand['tier'], (1, 999))
            if t_range[0] <= anchor['range'][1] and t_range[1] >= anchor['range'][0]:
                fam = get_family(cand['tier'])
                if fam not in family_champions or cand['score'] > family_champions[fam]['score']:
                    family_champions[fam] = {'tier': cand['tier'], 'score': cand['score'], 'id': cand['id']}

    for col in range(6):
        data = slot_matches.get(col)
        if not data: continue
        is_valid = False
        if data['status'] == 'empty_dna':
            detected, final = 'empty_dna', {'tier': 'empty_dna', 'score': 0.0}
        else:
            valid_options = [c for c in data['candidates'] if get_family(c['tier']) in family_champions]
            valid_options = [c for c in valid_options if c['tier'] == family_champions[get_family(c['tier'])]['tier']]
            valid_options.sort(key=lambda x: x['score'], reverse=True)
            if not valid_options:
                detected, final = 'low_conf_id', {'tier': 'none', 'score': 0.0}
            else:
                final = valid_options[0]
                for challenger in valid_options[1:]:
                    if 'dirt' in challenger['tier'] and challenger['score'] > (final['score'] - TIER_CONF_BUFFER):
                        final = challenger
                
                # ADAPTIVE GATE: Higher for Active, lower for Shadow
                gate = SHADOW_GATE if data['is_sha'] else ACTIVE_GATE
                is_valid = (final['score'] > gate) or (data['z_score'] > Z_TRUST_THRESHOLD and final['score'] > 0.35)
                detected = final['tier'] if is_valid else "low_conf_id"
        
        truth_tier = GROUND_TRUTH.get((f_idx, col))
        truth_rank = -1
        if truth_tier and truth_tier != 'empty_dna':
            for r, c in enumerate(data.get('candidates', [])):
                if c['tier'] == truth_tier: 
                    truth_rank = r + 1
                    break
        
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        if detected == 'empty_dna': color = (100, 100, 100)
        if col == anchor['col']: color = (0, 255, 255)
        
        cx, side_px = int(ORE0_X + (col * STEP)), int(DIM_ID * TARGET_SCALE)
        rx1, ry1 = int(cx - side_px//2), int(row4_y_base - side_px//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+side_px, ry1+side_px), color, 1)
        
        # Overlays
        draw_shadow_text(img_color, detected, (rx1 + 3, ry1 + side_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        y_stagger = 10 if col % 2 == 0 else 20
        stats_label = f"{final['score']:.2f} Z:{data.get('z_score', 0):.1f}"
        draw_shadow_text(img_color, stats_label, (rx1 + 3, ry1 + y_stagger), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        if truth_tier and detected != truth_tier:
            cv2.rectangle(img_color, (rx1-2, ry1-2), (rx1+side_px+2, ry1+side_px+2), (255, 0, 0), 1)
            draw_shadow_text(img_color, f"T:{truth_rank}", (rx1 + side_px - 25, ry1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        if is_valid: has_detections = True
        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'xhair': data['xhair'], 'truth_rank': truth_rank})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"resolver_v89_f{f_idx}.jpg"), img_color)
    return frame_results

def run_precision_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    truth_frames = list(set([k[0] for k in GROUND_TRUTH.keys()]))
    df_sample = df[df['frame_idx'].isin(truth_frames)]
    if len(df_sample) < 400:
        remaining = df[~df['frame_idx'].isin(truth_frames)].sample(min(400 - len(df_sample), len(df)))
        df_sample = pd.concat([df_sample, remaining])

    print(f"--- ORE ID AUDIT v8.9: THE ADAPTIVE CONTEXT RESOLVER ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, buffer_dir=buffer_dir)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v8.9_precision.csv"), index=False)
    
    print(f"\n--- PRECISION ERROR ANALYSIS ---")
    ores_only = audit_df[audit_df['truth_rank'] != -1]
    if not ores_only.empty:
        correct = len(ores_only[ores_only['detected'] == ores_only['truth_rank'].apply(lambda x: GROUND_TRUTH.get((ores_only.loc[ores_only.index[0], 'frame'], ores_only.loc[ores_only.index[0], 'slot'])))]) # Simplified check
        # More robust accuracy print
        print(f"Average Rank of True Ore: {ores_only['truth_rank'].mean():.1f}")
    
    print(f"\n--- DETECTION SUMMARY ---")
    print(audit_df['detected'].value_counts())
    print(f"\n--- DONE ---")

if __name__ == "__main__":
    run_precision_audit()