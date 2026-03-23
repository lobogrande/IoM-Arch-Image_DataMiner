# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using the Forensic Trinity:
#          Temporal Consensus integrated with v13.0 Diagnostic Logic.
# Version: 4.1 (Fixed NameError & Stabilized Resource Loading)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. GRID & HUD CONSTANTS ---
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

LIMIT_FLOORS = 20 # Set to None for full run

# --- 2. DATA-DRIVEN DIAGNOSTIC CONSTANTS ---
Z_TRUST_THRESHOLD = 2.0         
STATE_COMPLEXITY_THRESHOLD = 320 
LUMINANCE_SHADOW_FLOOR = 88      
MIN_FUSED_GATE = 0.28           
SHADOW_TEX_W, SHADOW_SIL_W, SHADOW_GRA_W = 0.35, 0.35, 0.30
ROTATION_VARIANTS = [-3, 3]

PLAYER_PRESENCE_GATE = 0.70 
HARVEST_COUNT = 15

BULLY_PENALTIES = {
    'epic1': 0.04, 'epic2': 0.04, 'epic3': 0.05,
    'leg1': 0.06, 'leg2': 0.06, 'leg3': 0.08,
    'myth1': 0.04, 'myth2': 0.05, 'myth3': 0.06,
    'div1': 0.12, 'div2': 0.12, 'div3': 0.15, 
    'com3': 0.03, 'dirt3': 0.03
}

CACHED_MASKS = {}

def get_cached_mask(exclusion_top):
    key = int(exclusion_top * 100)
    if key not in CACHED_MASKS:
        mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
        radius = int(18 * SCALE)
        cv2.circle(mask, (SIDE_PX//2, SIDE_PX//2), radius, 255, -1)
        if exclusion_top > 0:
            t_lim = int(SIDE_PX * exclusion_top)
            mask[0:t_lim, :] = 0
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

def get_gradient_map(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def get_complexity(img):
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def contains_vibrant_crosshair(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return False
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
        if cv2.countNonZero(mask) > 120:
            if cv2.mean(s, mask=mask)[0] > 110: return True
    return False

def load_all_templates():
    """Loads all templates including rotational variants for the Trinity sensors."""
    templates = {'active': {}, 'shadow': {}, 'player': []}
    t_path = cfg.TEMPLATE_DIR
    print(f"Loading Resources from {t_path}...")
    
    for f in os.listdir(t_path):
        if "negative_player" in f:
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None: templates['player'].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))
            continue
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "negative"]): continue
        
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        img_proc = apply_texture_enhancement(img_scaled, state)
        
        templates[state][tier].append({
            'img': img_proc, 'sil': get_silhouette_mask(img_proc), 'gra': get_gradient_map(img_proc)
        })
        for angle in ROTATION_VARIANTS:
            img_rot = rotate_image(img_proc, angle)
            templates[state][tier].append({
                'img': img_rot, 'sil': get_silhouette_mask(img_rot), 'gra': get_gradient_map(img_rot)
            })
    return templates

def check_side_slice_empty(roi_gray, is_banner):
    slot_48 = roi_gray[4:52, 4:52]
    slice_roi = slot_48[12:, 0:12] if is_banner else slot_48[:, 0:12]
    std_val = np.std(slice_roi)
    return std_val, std_val < 13.0

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score, best_roi_gray = 0.0, None
    frame_candidates = []

    for f_idx in f_range:
        img_path = os.path.join(buffer_dir, all_files[f_idx])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi_bgr.shape[:2] != (SIDE_PX, SIDE_PX): continue
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Player Obstruction Check
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt[13:43, 13:43], roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        comp = get_complexity(roi_gray)
        if best_roi_gray is None or comp > get_complexity(best_roi_gray):
            best_roi_gray = roi_gray
            
        if contains_vibrant_crosshair(roi_bgr): continue
        
        if max_p < 0.65:
            frame_candidates.append({'gray': roi_gray, 'comp': comp})

    tier_z_momentum = defaultdict(float)
    max_z_seen, best_overall_score = 0.0, 0.0
    top_frames = sorted(frame_candidates, key=lambda x: x['comp'], reverse=True)[:HARVEST_COUNT]

    for f in top_frames:
        roi_gray = f['gray']
        roi_comp = f['comp']
        mean_lum = np.mean(roi_gray)
        
        target_state = 'active' if (roi_comp > STATE_COMPLEXITY_THRESHOLD or mean_lum > LUMINANCE_SHADOW_FLOOR) else 'shadow'
        roi_proc = apply_texture_enhancement(roi_gray, target_state)
        roi_sil = get_silhouette_mask(roi_proc)
        roi_gra = get_gradient_map(roi_proc)
        
        active_mask = get_cached_mask(0.40 if is_banner else 0.0)
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            
            best_t_score = 0
            for tpl in res[target_state][tier]:
                if target_state == 'active':
                    s = cv2.minMaxLoc(cv2.matchTemplate(roi_proc, tpl['img'], cv2.TM_CCORR_NORMED, mask=active_mask))[1]
                    if s > best_t_score: best_t_score = s
                else:
                    s_tex = cv2.minMaxLoc(cv2.matchTemplate(roi_proc, tpl['img'], cv2.TM_CCOEFF_NORMED))[1]
                    s_sil = cv2.minMaxLoc(cv2.matchTemplate(roi_sil, tpl['sil'], cv2.TM_CCOEFF_NORMED))[1]
                    s_gra = cv2.minMaxLoc(cv2.matchTemplate(roi_gra, tpl['gra'], cv2.TM_CCOEFF_NORMED))[1]
                    fused = (s_tex * SHADOW_TEX_W) + (s_sil * SHADOW_SIL_W) + (s_gra * SHADOW_GRA_W)
                    if fused > best_t_score: best_t_score = fused
            
            frame_results.append({'tier': tier, 'score': best_t_score - penalty})

        if not frame_results: continue
        scores = np.array([x['score'] for x in frame_results])
        if len(scores) > 1:
            mean_s, std_s = np.mean(scores), np.std(scores)
            winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
            z_score = (winner['score'] - mean_s) / max(0.01, std_s)
            
            if z_score > 1.0 or winner['score'] > MIN_FUSED_GATE:
                tier_z_momentum[winner['tier']] += max(1.0, z_score)
                max_z_seen = max(max_z_seen, z_score)
                best_overall_score = max(best_overall_score, winner['score'])

    if tier_z_momentum:
        winner_tier = max(tier_z_momentum, key=tier_z_momentum.get)
        gate = 0.40 if any(f in winner_tier for f in ['dirt', 'com']) else 0.46
        if best_overall_score > gate or max_z_seen > Z_TRUST_THRESHOLD:
            tag = "[Z]" if max_z_seen > Z_TRUST_THRESHOLD else "[M]"
            return winner_tier, round(best_overall_score, 4), int(tier_z_momentum[winner_tier]), peak_p_score, tag

    # STRICT FORENSIC FALLBACK
    if peak_p_score >= PLAYER_PRESENCE_GATE and best_roi_gray is not None:
        val, is_empty = check_side_slice_empty(best_roi_gray, is_banner)
        if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    return "low_conf", round(best_overall_score, 4), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    if hasattr(cfg, 'BOSS_DATA') and f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_tag"] = "[B]"
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1)
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key], results[f"{key}_tag"] = "empty", ""
            else:
                tier, score, momentum, pmax, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_score"], results[f"{key}_mom"], results[f"{key}_pmax"], results[f"{key}_tag"] = tier, score, momentum, pmax, tag
    return results

def run_tier_identification():
    print(f"--- STEP 4.2: TIER IDENTIFICATION v4.1 ---")
    if not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: {DNA_INVENTORY_CSV} not found.")
        return
        
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    
    if LIMIT_FLOORS:
        df_floors = df_floors.head(LIMIT_FLOORS)
        
    buffer_dir = cfg.get_buffer_path(0)
    res = load_all_templates()
    
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    if not os.path.exists(VERIFY_DIR):
        os.makedirs(VERIFY_DIR)
    
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Success: {tag_counts['[Z]']}+{tag_counts['[M]']}, Likely: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    
    # Generate Visual Proofs
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                
                # Color Coding
                if tier == "likely_empty": color = (0, 255, 255) # Cyan
                elif tier == "low_conf": color = (0, 0, 255)     # Red
                else: color = (0, 255, 0)                        # Green
                
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, color, 1)
                
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()