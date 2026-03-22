# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 12.8 (The Forensic Scalpel: Clean-Room Deduplication & Rank Agreement)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
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
ROTATION_VARIANTS = [-3, 3] 

# LOGIC THRESHOLDS
Z_TRUST_THRESHOLD = 2.1 
STATE_COMPLEXITY_THRESHOLD = 320 
LUMINANCE_SHADOW_FLOOR = 88      
TIER_CONF_BUFFER = 0.08 

# DUAL-SENSOR CONSTANTS (v12.8 Refined)
MIN_SILHOUETTE_GATE = 0.50     
MIN_TEXTURE_GATE = 0.18        

BULLY_PENALTIES = {
    'rare1': 0.05, # v12.8: Added to prevent rare1 from out-matching high tiers
    'epic1': 0.08, 'epic2': 0.08, 'epic3': 0.10,
    'leg1': 0.12, 'leg2': 0.12, 'leg3': 0.14,
    'myth1': 0.05, 'myth2': 0.08, 'myth3': 0.10,
    'div1': 0.20, 'div2': 0.20, 'div3': 0.25, 
    'com3': 0.05, 'dirt3': 0.03
}

# Pre-cached masks
CACHED_MASKS = {}

def get_cached_mask(exclusion_top, exclusion_bot=0.0):
    key = (int(exclusion_top * 100), int(exclusion_bot * 100))
    if key not in CACHED_MASKS:
        mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
        radius = int(18 * (SIDE_PX / 48))
        cv2.circle(mask, (SIDE_PX//2, SIDE_PX//2), radius, 255, -1)
        t_lim = int(SIDE_PX * exclusion_top)
        mask[0:t_lim, :] = 0
        if exclusion_bot > 0:
            b_lim = int(SIDE_PX * (1.0 - exclusion_bot))
            mask[b_lim:SIDE_PX, :] = 0
        CACHED_MASKS[key] = mask
    return CACHED_MASKS[key]

def apply_texture_enhancement(img, state='active'):
    if img is None or img.size == 0: return img
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clip = 3.0 if state == 'shadow' else 1.8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    return clahe.apply(normalized)

def get_silhouette_mask(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh[SIDE_PX//2, SIDE_PX//2] == 0: thresh = cv2.bitwise_not(thresh)
    return thresh

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def get_complexity(img):
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def detect_vibrant_crosshair(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return "none", 0, 0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ranges = {
        'GOLD': ([12, 85, 85], [42, 255, 255]),
        'BLUE': ([95, 85, 85], [140, 255, 255]),
        'RED':  ([0, 85, 85], [10, 255, 255]),
        'RED2': ([165, 85, 85], [180, 255, 255])
    }
    for name, (low, high) in ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        px_count = cv2.countNonZero(mask)
        if px_count > 180: 
            mean_sat = cv2.mean(s, mask=mask)[0]
            if mean_sat > 110: return name.rstrip('2'), px_count, round(mean_sat, 1)
    return "none", 0, 0

def load_all_templates():
    templates = {'active': {}, 'shadow': {}}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates
    
    print(f"Loading Multi-Sensor Standards (Plain Ores)...")
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        if "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "player", "negative", "pickaxe"]): continue
        
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        img_proc = apply_texture_enhancement(img_scaled, state)
        img_sil = get_silhouette_mask(img_proc)
        
        templates[state][tier].append({
            'id': f, 'img': img_proc, 'sil': img_sil, 'angle': 0, 'tier': tier, 'comp': get_complexity(img_proc)
        })
        for angle in ROTATION_VARIANTS:
            img_rot = rotate_image(img_proc, angle)
            templates[state][tier].append({
                'id': f, 'img': img_rot, 'sil': get_silhouette_mask(img_rot), 'angle': angle, 'tier': tier, 'comp': get_complexity(img_rot)
            })
    
    active_tiers = set(templates['active'].keys())
    shadow_tiers = set(templates['shadow'].keys())
    print(f"  [ACTIVE] Loaded {len(active_tiers)} distinct tiers.")
    print(f"  [SHADOW] Loaded {len(shadow_tiers)} distinct tiers.")
    missing = active_tiers - shadow_tiers
    if missing: print(f"  WARNING: Shadow state is missing templates for: {', '.join(missing)}")
        
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
    
    frame_results = []
    has_detections = False

    for col in range(6):
        if r4_dna[col] == '0':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_dna', 'score': 0.0, 'xhair': 'none'})
            continue
            
        cx = int(ORE0_X + (col * STEP))
        tx1, ty1 = int(cx - SIDE_PX//2), int(row4_y_base - SIDE_PX//2)
        if ty1 < 0 or ty1+SIDE_PX > img_color.shape[0] or tx1 < 0 or tx1+SIDE_PX > img_color.shape[1]:
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'low_conf_id', 'score': 0.0, 'xhair': 'none'})
            continue

        roi_bgr = img_color[ty1:ty1+SIDE_PX, tx1:tx1+SIDE_PX]
        roi_gray = img_gray[ty1:ty1+SIDE_PX, tx1:tx1+SIDE_PX]
        
        xhair, _, _ = detect_vibrant_crosshair(roi_bgr)
        if xhair != 'none':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'xhair_obscured', 'score': 0.0, 'xhair': xhair})
            continue

        mean_lum = np.mean(roi_gray)
        roi_norm = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        roi_comp = get_complexity(roi_norm)
        target_state = 'active' if (roi_comp > STATE_COMPLEXITY_THRESHOLD or mean_lum > LUMINANCE_SHADOW_FLOOR) else 'shadow'
        
        roi_proc = apply_texture_enhancement(roi_gray, target_state)
        roi_sil = get_silhouette_mask(roi_proc)
        
        top_half, bot_half = roi_gray[0:SIDE_PX//2, :], roi_gray[SIDE_PX//2:SIDE_PX, :]
        bot_e = get_complexity(bot_half)
        active_mask = get_cached_mask(0.40, 0.12 if bot_e > 2000 else 0.0)
        
        texture_cands = []
        sil_cands = []
        
        for tier, variants in templates[target_state].items():
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            for tpl in variants:
                # Texture Sensor
                res_tex = cv2.matchTemplate(roi_proc, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=active_mask)
                _, s_tex, _, _ = cv2.minMaxLoc(res_tex)
                texture_cands.append({'tier': tier, 'score': s_tex - penalty})
                
                # Geometry Sensor
                res_sil = cv2.matchTemplate(roi_sil, tpl['sil'], cv2.TM_CCOEFF_NORMED)
                _, s_sil, _, _ = cv2.minMaxLoc(res_sil)
                sil_cands.append({'tier': tier, 'score': s_sil})

        # v12.8: CLEAN-ROOM DEDUPLICATION
        # Consolidate rotation variants into unique tiers BEFORE ranking
        t_df = pd.DataFrame(texture_cands).groupby('tier')['score'].max().reset_index().sort_values('score', ascending=False)
        s_df = pd.DataFrame(sil_cands).groupby('tier')['score'].max().reset_index().sort_values('score', ascending=False)
        
        top_texture = t_df.iloc[0]
        top_sil = s_df.iloc[0]
        
        is_valid = False
        detected = 'low_conf_id'
        final_score = top_texture['score']
        
        if target_state == 'active':
            # v12.8: Measure Uniqueness against deduplicated tiers
            z = (top_texture['score'] - t_df['score'].mean()) / max(1e-6, t_df['score'].std())
            gate = 0.40 if any(f in top_texture['tier'] for f in ['dirt', 'com']) else 0.48
            
            is_valid = (top_texture['score'] > gate) or (z > 2.2 and top_texture['score'] > 0.20)
            detected = top_texture['tier'] if is_valid else 'low_conf_id'
        else:
            # v12.8: Shadow Handshake (Rank-1 Agreement)
            if top_texture['tier'] == top_sil['tier']:
                # Rank 1 Agreement is a very strong signal
                if top_texture['score'] > MIN_TEXTURE_GATE and top_sil['score'] > MIN_SILHOUETTE_GATE:
                    detected = f"{top_texture['tier']}[S]"
                    is_valid = True
                    final_score = (top_texture['score'] + top_sil['score']) / 2
            else:
                # Top-3 Handshake fallback
                t_list = t_df.head(3)['tier'].tolist()
                s_list = s_df.head(3)['tier'].tolist()
                common = [t for t in t_list if t in s_list]
                if common:
                    best_tier = common[0]
                    t_val = t_df[t_df['tier'] == best_tier]['score'].iloc[0]
                    s_val = s_df[s_df['tier'] == best_tier]['score'].iloc[0]
                    if t_val > 0.25 and s_val > 0.60: # Higher bar for non-Rank-1 agreement
                        detected = f"{best_tier}[S]"
                        is_valid = True
                        final_score = (t_val + s_val) / 2

        color = (0, 255, 0) if is_valid else (0, 0, 255)
        rx1, ry1 = int(ORE0_X + (col * STEP) - SIDE_PX//2), int(row4_y_base - SIDE_PX//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+SIDE_PX, ry1+SIDE_PX), color, 1)
        draw_shadow_text(img_color, detected, (rx1+3, ry1+SIDE_PX-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        stat_str = f"{final_score:.2f} C:{int(roi_comp)}"
        draw_shadow_text(img_color, stat_str, (rx1+3, ry1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        
        if is_valid: has_detections = True
        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final_score, 4), 'xhair': xhair})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"scalpel_v128_f{f_idx}.jpg"), img_color)
    return frame_results

def run_precision_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df_sample = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v12.8: THE FORENSIC SCALPEL ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, buffer_dir=buffer_dir)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_func, r): r for r in df_sample.to_dict('records')[:500]}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = f.result()
                if res: all_results.extend(res)
                if i % 100 == 0: print(f"  Processed {i} frames...")
            except Exception as e: print(f"  Worker Error: {e}")
    
    if all_results:
        audit_df = pd.DataFrame(all_results)
        audit_path = os.path.join(OUT_DIR, "ore_id_v12.8_precision.csv")
        audit_df.to_csv(audit_path, index=False)
        print(f"\nSaved CSV to: {audit_path}")
        print(f"--- DETECTION SUMMARY ---\n{audit_df['detected'].value_counts()}")

if __name__ == "__main__":
    run_precision_audit()