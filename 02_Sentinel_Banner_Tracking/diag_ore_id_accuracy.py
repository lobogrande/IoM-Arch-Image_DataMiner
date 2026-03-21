# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 9.4 (The Optical Shield: Vibrancy Sensing & Dynamic Mod-Blackout)

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
TARGET_SCALE = 1.20
ROW4_Y_PERSPECTIVE_SHIFT = 2 
ROTATION_VARIANTS = [-3, 0, 3] 

# LOGIC THRESHOLDS (Based on v1.2 Forensics)
Z_TRUST_THRESHOLD = 2.1 
DIRT_COMPLEXITY_CEILING = 600   # Dirt cannot be this "busy"
MOD_ENERGY_RATIO_TRIGGER = 1.4  # Trigger for aggressive masking
XHAIR_PX_FLOOR = 150            # Minimum pixels for a real ring
XHAIR_SAT_FLOOR = 130           # Minimum vibrancy for a real ring
TIER_CONF_BUFFER = 0.08 

BULLY_PENALTIES = {
    'leg1': 0.08, 'leg2': 0.06, 'leg3': 0.10,
    'myth1': 0.05, 'myth2': 0.08, 'myth3': 0.10,
    'div3': 0.15, 'com3': 0.05
}

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def apply_gamma_lift(img, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def get_complexity(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_spatial_mask(dim, exclusion_percent=0.40):
    mask = np.zeros((dim, dim), dtype=np.uint8)
    radius = int(18 * (dim / 48))
    cv2.circle(mask, (dim//2, dim//2), radius, 255, -1)
    top_limit = int(dim * exclusion_percent)
    mask[0:top_limit, :] = 0
    return mask

def detect_vibrant_crosshair(roi_bgr):
    """Detects targeting rings using Forensic v1.2 Vibrancy rules."""
    if roi_bgr is None or roi_bgr.size == 0: return "none", 0, 0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    ranges = {
        'GOLD': ([12, 100, 100], [42, 255, 255]),
        'BLUE': ([95, 100, 100], [140, 255, 255]),
        'RED':  ([0, 100, 100], [10, 255, 255]),
        'RED_EXT': ([165, 100, 100], [180, 255, 255])
    }
    
    for name, (low, high) in ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        px_count = cv2.countNonZero(mask)
        if px_count > XHAIR_PX_FLOOR:
            mean_sat = cv2.mean(s, mask=mask)[0]
            if mean_sat > XHAIR_SAT_FLOOR:
                return name.rstrip('EXT'), px_count, round(mean_sat, 1)
    return "none", 0, 0

def load_all_templates():
    templates = {'ore_id': {}}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')) or f.startswith(("background", "negative")): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        if tier not in templates['ore_id']: templates['ore_id'][tier] = []
        new_dim = int(DIM_ID * TARGET_SCALE)
        img_scaled = cv2.resize(img_raw, (new_dim, new_dim), interpolation=cv2.INTER_AREA)
        for angle in ROTATION_VARIANTS:
            img_rot = rotate_image(img_scaled, angle) if angle != 0 else img_scaled
            templates['ore_id'][tier].append({
                'id': f, 'img': img_rot, 'angle': angle, 'tier': tier,
                'comp': get_complexity(apply_gamma_lift(img_rot, 0.6))
            })
    return templates

def draw_shadow_text(img, text, pos, font, scale, color, thick):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale, (0,0,0), thick+1)
    cv2.putText(img, text, pos, font, scale, color, thick)

def process_single_frame(frame_data, dna_map, templates, buffer_dir):
    f_idx = frame_data['frame_idx']
    img_color = cv2.imread(os.path.join(buffer_dir, frame_data['filename']))
    if img_color is None: return []
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y_base = int(ORE0_Y + (3 * STEP)) + ROW4_Y_PERSPECTIVE_SHIFT
    side_px = int(DIM_ID * TARGET_SCALE)
    
    slot_matches = {}
    for col in range(6):
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna', 'candidates': []}
            continue
            
        cx = int(ORE0_X + (col * STEP))
        tx1, ty1 = int(cx - side_px//2), int(row4_y_base - side_px//2)
        roi_bgr = img_color[ty1:ty1+side_px, tx1:tx1+side_px]
        roi_gray = img_gray[ty1:ty1+side_px, tx1:tx1+side_px]
        
        # 1. Forensic Sensor Pass
        xhair, xh_px, xh_sat = detect_vibrant_crosshair(roi_bgr)
        top_half, bot_half = roi_gray[0:side_px//2, :], roi_gray[side_px//2:side_px, :]
        top_e, bot_e = get_complexity(top_half), get_complexity(bot_half)
        ratio = top_e / max(1, bot_e)
        
        # 2. Dynamic Masking: If mod noise is high, blackout the top 60%
        mask_lvl = 0.60 if ratio > MOD_ENERGY_RATIO_TRIGGER else 0.40
        active_mask = get_spatial_mask(side_px, mask_lvl)
        
        # 3. Match
        roi_lifted = apply_gamma_lift(roi_gray, 0.5)
        roi_comp = get_complexity(roi_lifted)
        all_candidates = []
        
        for tier, variants in templates['ore_id'].items():
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            # HARD FENCE: If ROI is too complex, it cannot be DIRT
            if 'dirt' in tier and roi_comp > DIRT_COMPLEXITY_CEILING: continue
            
            for tpl in variants:
                side = tpl['img'].shape[0]
                x1, y1 = int(cx - (side//2) - X_JITTER), int(row4_y_base - (side//2) - Y_JITTER)
                search_area = img_gray[y1:y1+side+(Y_JITTER*2), x1:x1+side+(X_JITTER*2)]
                if search_area.shape[0] < side or search_area.shape[1] < side: continue
                search_lifted = apply_gamma_lift(search_area, 0.5)
                
                res = cv2.matchTemplate(search_lifted, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=active_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                
                # Complexity Affinity
                affinity = 0.035 if abs(tpl['comp'] - roi_comp) < 15.0 else 0
                all_candidates.append({'tier': tier, 'score': score - penalty + affinity, 'angle': tpl['angle']})
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        scores = [c['score'] for c in all_candidates]
        z_score = (scores[0] - np.mean(scores)) / np.std(scores) if len(scores) > 1 else 0
        
        slot_matches[col] = {
            'status': 'occupied', 'candidates': all_candidates, 'z_score': z_score,
            'xhair': xhair, 'xhair_px': xh_px, 'xhair_sat': xh_sat, 'ratio': ratio, 'roi_comp': roi_comp
        }

    # --- CONSENSUS ---
    floor_votes = []
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna' or not data['candidates']: continue
        top = data['candidates'][0]
        if top['score'] > 0.45 or data['z_score'] > Z_TRUST_THRESHOLD:
            floor_votes.append(ORE_RESTRICTIONS.get(top['tier'], (1, 999)))
    
    consensus_range = Counter(floor_votes).most_common(1)[0][0] if floor_votes else (1, 999)
    family_champions = {}
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna': continue
        for cand in data['candidates']:
            t_range = ORE_RESTRICTIONS.get(cand['tier'], (1, 999))
            if t_range[0] <= consensus_range[1] and t_range[1] >= consensus_range[0]:
                fam = ''.join([i for i in cand['tier'] if not i.isdigit()])
                if fam not in family_champions or cand['score'] > family_champions[fam]['score']:
                    family_champions[fam] = {'tier': cand['tier'], 'score': cand['score']}

    # --- RESOLUTION ---
    frame_results = []
    has_detections = False
    for col in range(6):
        data = slot_matches[col]
        is_valid, detected, final = False, 'none', {'tier': 'none', 'score': 0.0, 'angle': 0}
        
        if data['status'] == 'empty_dna':
            detected = 'empty_dna'
        else:
            valid_opts = [c for c in data['candidates'] if c['tier'] in [v['tier'] for v in family_champions.values()]]
            if valid_opts:
                valid_opts.sort(key=lambda x: x['score'], reverse=True)
                final = valid_opts[0]
                # Dirt Bias
                for ch in valid_opts[1:]:
                    if 'dirt' in ch['tier'] and ch['score'] > (final['score'] - TIER_CONF_BUFFER): final = ch
                
                gate = 0.55 if 'dirt' in final['tier'] else 0.64
                is_valid = (final['score'] > gate) or (data['z_score'] > Z_TRUST_THRESHOLD and final['score'] > 0.22)
                
                if is_valid:
                    if data['xhair'] != 'none':
                        detected = "xhair_obscured"
                        is_valid = False
                    else:
                        detected = final['tier']

        # Visuals
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        if detected == 'empty_dna': color = (100, 100, 100)
        elif detected == 'xhair_obscured': color = (0, 255, 255)
        
        cx, side_px = int(ORE0_X + (col * STEP)), int(DIM_ID * TARGET_SCALE)
        rx1, ry1 = int(cx - side_px//2), int(row4_y_base - side_px//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+side_px, ry1+side_px), color, 1)
        
        y_label = ry1 + side_px - (5 if col % 2 == 0 else 15)
        draw_shadow_text(img_color, detected, (rx1+3, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        y_stats = ry1 + (10 if col % 2 == 0 else 22)
        stat_str = f"{final['score']:.2f} Z:{data.get('z_score',0):.1f}"
        if data.get('xhair') != 'none': stat_str += f" S:{data['xhair_sat']}"
        draw_shadow_text(img_color, stat_str, (rx1+3, y_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        
        if is_valid: has_detections = True
        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'xhair': data.get('xhair','none')})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"optical_v94_f{f_idx}.jpg"), img_color)
    return frame_results

def run_precision_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df_sample = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v9.4: THE OPTICAL SHIELD ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, buffer_dir=buffer_dir)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_func, r): r for r in df_sample.to_dict('records')[:1000]}
        for f in concurrent.futures.as_completed(futures): all_results.extend(f.result())
    
    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v9.4_precision.csv"), index=False)
    print(f"\n--- DETECTION SUMMARY ---\n{audit_df['detected'].value_counts()}\n--- DONE ---")

if __name__ == "__main__":
    run_precision_audit()