# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 9.6 (The Forensic Guard: Two-Pass Optimization & Tier Isolation)

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

# ROI CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
DIM_ID = 48
TARGET_SCALE = 1.20
SIDE_PX = int(DIM_ID * TARGET_SCALE)
ROTATION_VARIANTS = [-3, 3] # Secondary variants for damage animations

# LOGIC THRESHOLDS
Z_TRUST_THRESHOLD = 2.1 
DIRT_COMPLEXITY_CEILING = 600    # No Dirt/Common allowed above this
HIGH_TIER_COMPLEXITY_FLOOR = 400 # No Div/Leg/Myth/Epic allowed below this
MOD_ENERGY_RATIO_TRIGGER = 1.4 
XHAIR_PX_FLOOR = 200            
XHAIR_SAT_FLOOR = 140           
TIER_CONF_BUFFER = 0.08 

# Pre-cached masks
CACHED_MASKS = {}

def get_cached_mask(exclusion_top, exclusion_bot=0.0):
    key = (int(exclusion_top * 100), int(exclusion_bot * 100))
    if key not in CACHED_MASKS:
        mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
        radius = int(18 * (SIDE_PX / 48))
        cv2.circle(mask, (SIDE_PX//2, SIDE_PX//2), radius, 255, -1)
        # Top mask (mods)
        t_lim = int(SIDE_PX * exclusion_top)
        mask[0:t_lim, :] = 0
        # Bottom mask (health bars)
        if exclusion_bot > 0:
            b_lim = int(SIDE_PX * (1.0 - exclusion_bot))
            mask[b_lim:SIDE_PX, :] = 0
        CACHED_MASKS[key] = mask
    return CACHED_MASKS[key]

BULLY_PENALTIES = {
    'leg1': 0.10, 'leg2': 0.08, 'leg3': 0.12,
    'myth1': 0.05, 'myth2': 0.08, 'myth3': 0.10,
    'div1': 0.15, 'div2': 0.15, 'div3': 0.20, 
    'com3': 0.05
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
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def detect_vibrant_crosshair(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return "none", 0, 0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ranges = {
        'GOLD': ([12, 110, 110], [42, 255, 255]),
        'BLUE': ([95, 110, 110], [140, 255, 255]),
        'RED':  ([0, 110, 110], [10, 255, 255]),
        'RED2': ([165, 110, 110], [180, 255, 255])
    }
    for name, (low, high) in ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        px_count = cv2.countNonZero(mask)
        if px_count > XHAIR_PX_FLOOR:
            mean_sat = cv2.mean(s, mask=mask)[0]
            if mean_sat > XHAIR_SAT_FLOOR:
                return name.rstrip('2'), px_count, round(mean_sat, 1)
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
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        
        # Primary template (0 degrees)
        templates['ore_id'][tier].append({
            'id': f, 'img': img_scaled, 'angle': 0, 'tier': tier,
            'comp': get_complexity(apply_gamma_lift(img_scaled, 0.6))
        })
        # Variants for two-pass check
        for angle in ROTATION_VARIANTS:
            img_rot = rotate_image(img_scaled, angle)
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
    row4_y_base = int(ORE0_Y + (3 * STEP)) + 2
    
    slot_matches = {}
    for col in range(6):
        # Always initialize keys to avoid terminal spam
        default_data = {
            'status': 'empty_dna', 'candidates': [], 'z_score': 0,
            'xhair': 'none', 'xhair_sat': 0, 'xhair_px': 0, 'ratio': 0, 'roi_comp': 0
        }
        
        if r4_dna[col] == '0':
            slot_matches[col] = default_data
            continue
            
        cx = int(ORE0_X + (col * STEP))
        tx1, ty1 = int(cx - SIDE_PX//2), int(row4_y_base - SIDE_PX//2)
        roi_bgr = img_color[ty1:ty1+SIDE_PX, tx1:tx1+SIDE_PX]
        roi_gray = img_gray[ty1:ty1+SIDE_PX, tx1:tx1+SIDE_PX]
        
        xhair, xh_px, xh_sat = detect_vibrant_crosshair(roi_bgr)
        top_half, bot_half = roi_gray[0:SIDE_PX//2, :], roi_gray[SIDE_PX//2:SIDE_PX, :]
        top_e, bot_e = get_complexity(top_half), get_complexity(bot_half)
        ratio = top_e / max(1, bot_e)
        
        # 1. Health Bar & Damage protection
        # Force aggressive masks if ROI shows "Hit" signals
        mask_top = 0.60 if (ratio > MOD_ENERGY_RATIO_TRIGGER or top_e > 2500) else 0.40
        mask_bot = 0.12 if (bot_e > 2000) else 0.0 # Ignore health bars if bottom is noisy
        active_mask = get_cached_mask(mask_top, mask_bot)
        
        roi_lifted = apply_gamma_lift(roi_gray, 0.5)
        roi_comp = get_complexity(roi_lifted)
        all_candidates = []
        
        # Determine if we need the full sweep or just primary
        is_hit_frame = (top_e > 1500 or bot_e > 1500)
        
        for tier, variants in templates['ore_id'].items():
            # STRUCTURAL GUARD (v9.6 HARD FENCING)
            is_complex_tier = any(x in tier for x in ['div', 'leg', 'myth', 'epic'])
            is_simple_tier = any(x in tier for x in ['dirt', 'com', 'rare'])
            
            if is_complex_tier and roi_comp < HIGH_TIER_COMPLEXITY_FLOOR: continue
            if is_simple_tier and roi_comp > DIRT_COMPLEXITY_CEILING: continue
            
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            for tpl in variants:
                # Two-pass performance optimization
                if not is_hit_frame and tpl['angle'] != 0: continue
                
                x1, y1 = int(cx - (SIDE_PX//2) - 2), int(row4_y_base - (SIDE_PX//2) - 1)
                search_area = img_gray[y1:y1+SIDE_PX+2, x1:x1+SIDE_PX+4]
                if search_area.shape[0] < SIDE_PX or search_area.shape[1] < SIDE_PX: continue
                search_lifted = apply_gamma_lift(search_area, 0.5)
                
                res = cv2.matchTemplate(search_lifted, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=active_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                
                affinity = 0.035 if abs(tpl['comp'] - roi_comp) < 15.0 else 0
                all_candidates.append({'tier': tier, 'score': score - penalty + affinity, 'angle': tpl['angle']})
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        scores = [c['score'] for c in all_candidates]
        z_score = (scores[0] - np.mean(scores)) / np.std(scores) if len(scores) > 1 else 0
        
        slot_matches[col] = {
            'status': 'occupied', 'candidates': all_candidates, 'z_score': z_score,
            'xhair': xhair, 'xhair_sat': xh_sat, 'xhair_px': xh_px, 'ratio': ratio, 'roi_comp': roi_comp
        }

    # Consensus Voting
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

    # Resolution
    frame_results = []
    has_detections = False
    for col in range(6):
        data = slot_matches[col]
        is_valid, detected, final = False, 'none', {'tier': 'none', 'score': 0.0}
        
        if data['status'] == 'empty_dna':
            detected = 'empty_dna'
        else:
            valid_opts = [c for c in data['candidates'] if c['tier'] in [v['tier'] for v in family_champions.values()]]
            if valid_opts:
                valid_opts.sort(key=lambda x: x['score'], reverse=True)
                final = valid_opts[0]
                for ch in valid_opts[1:]:
                    if 'dirt' in ch['tier'] and ch['score'] > (final['score'] - TIER_CONF_BUFFER): final = ch
                
                gate = 0.48 if 'dirt' in final['tier'] else 0.60
                is_valid = (final['score'] > gate) or (data['z_score'] > Z_TRUST_THRESHOLD and final['score'] > 0.22)
                if is_valid:
                    if data['xhair'] != 'none':
                        detected, is_valid = "xhair_obscured", False
                    else:
                        detected = final['tier']

        color = (0, 255, 0) if is_valid else (0, 0, 255)
        if detected == 'empty_dna': color = (100, 100, 100)
        elif detected == 'xhair_obscured': color = (0, 255, 255)
        
        rx1, ry1 = int(ORE0_X + (col * STEP) - SIDE_PX//2), int(row4_y_base - SIDE_PX//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+SIDE_PX, ry1+SIDE_PX), color, 1)
        draw_shadow_text(img_color, detected, (rx1+3, ry1+SIDE_PX-(5 if col%2==0 else 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        stat_str = f"{final['score']:.2f} Z:{data.get('z_score',0):.1f}"
        if data.get('xhair') != 'none': stat_str += f" S:{data['xhair_sat']}"
        draw_shadow_text(img_color, stat_str, (rx1+3, ry1+(10 if col%2==0 else 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        
        if is_valid: has_detections = True
        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'xhair': data.get('xhair','none')})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"structural_v96_f{f_idx}.jpg"), img_color)
    return frame_results

def run_precision_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df_sample = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v9.6: THE FORENSIC GUARD ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Increased sample to 2000 now that it's optimized
        futures = {executor.submit(worker_func, r): r for r in df_sample.to_dict('records')[:2000]}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = f.result()
                if res: all_results.extend(res)
                if i % 100 == 0: print(f"  Processed {i} frames...")
            except Exception as e:
                print(f"  Critical Error in Process: {e}")
    
    if all_results:
        audit_df = pd.DataFrame(all_results)
        audit_path = os.path.join(OUT_DIR, "ore_id_v9.6_precision.csv")
        audit_df.to_csv(audit_path, index=False)
        print(f"\nSaved CSV to: {audit_path}")
        print(f"--- DETECTION SUMMARY ---\n{audit_df['detected'].value_counts()}")
    else:
        print("No results generated.")
    print(f"--- DONE ---")

if __name__ == "__main__":
    run_precision_audit()