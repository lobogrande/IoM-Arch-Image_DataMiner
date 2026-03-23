# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using the Forensic Trinity:
#          Triple-Sensor Fusion (Texture, Geometry, Grain) and Structural Affinity.
# Version: 2.9 (The Forensic Trinity Engine)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. GRID & HUD CONSTANTS (Verified) ---
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

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = 20  # Set to None for production

# --- 2. TRINITY SENSOR CONSTANTS ---
# Weights for Fusion: Texture (40%), Geometry (30%), Grain (30%)
W_TEX, W_GEO, W_GRA = 0.40, 0.30, 0.30
MIN_FUSED_CONFIDENCE = 0.35  # Trinity agreement is more restrictive
AFFINITY_SIGMA = 0.40        # Stricter complexity envelope
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_WIDTH = 14        
SIDE_SLICE_STD_MAX = 14.0    
HARVEST_COUNT = 15          
COMPLEXITY_GATE = 300        

def get_complexity(img):
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def apply_clahe(img):
    """Normalizes contrast to expose structural grain."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def get_silhouette(img_gray):
    """Produces binary geometry map for shape matching."""
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure ore (center) is white
    if thresh[SIDE_PX//2, SIDE_PX//2] == 0: thresh = cv2.bitwise_not(thresh)
    return thresh

def get_gradient_map(img_gray):
    """Produces Sobel-based edge magnitude map for grain matching."""
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_resources():
    """Pre-calculates Trinity Blueprints (Texture/Geo/Grain) for all templates."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    print("Blueprinting Trinity Template Library...")
    
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_scaled = cv2.resize(img, (SIDE_PX, SIDE_PX))
        
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = {'tpls': [], 'mean_comp': 0.0}
            
            # Create Trinity blueprints
            img_tex = apply_clahe(img_scaled)
            img_geo = get_silhouette(img_tex)
            img_gra = get_gradient_map(img_tex)
            
            res['ores'][tier]['tpls'].append({
                'tex': img_tex, 'geo': img_geo, 'gra': img_gra,
                'comp': get_complexity(img_scaled)
            })
            
        if "negative_player" in f: res['player'].append(img_scaled)
        if "background_plain" in f: res['bg'].append(img_scaled)

    for tier in res['ores']:
        comps = [t['comp'] for t in res['ores'][tier]['tpls']]
        res['ores'][tier]['mean_comp'] = np.mean(comps)
        
    return res

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """Extreme-Edge Forensic check for background presence."""
    slot_48 = roi_gray[4:52, 4:52]
    if is_banner: slot_48 = slot_48[12:, :]
    slice_roi = slot_48[:, 0:SIDE_SLICE_WIDTH]
    if np.std(slice_roi) < SIDE_SLICE_STD_MAX: return np.std(slice_roi), True
    best_s = 0
    for tpl in bg_tpls:
        tpl_slice = tpl[:, 0:SIDE_SLICE_WIDTH]
        if is_banner: tpl_slice = tpl_slice[12:, :]
        res = cv2.matchTemplate(tpl_slice, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
    return best_s, best_s > 0.70

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Weighted consensus engine driven by Triple-Sensor Fusion."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    peak_p_score, last_roi_gray = 0.0, None

    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_gray = cv2.cvtColor(img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX], cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        last_roi_gray = roi_gray
        # Player Obstruction Check
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        if max_p < PLAYER_REJECTION_GATE:
            comp = get_complexity(roi_gray)
            if comp >= COMPLEXITY_GATE:
                frame_candidates.append({'gray': roi_gray, 'comp': comp})

    if not frame_candidates:
        if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
            val, is_empty = check_side_slice_empty(last_roi_gray, res['bg'], is_banner)
            if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"
        return "low_conf", 0.0, 0, peak_p_score, ""

    tier_momentum = defaultdict(float)
    best_overall_score = 0
    top_frames = sorted(frame_candidates, key=lambda x: x['comp'], reverse=True)[:HARVEST_COUNT]
    
    for f in top_frames:
        roi_gray = f['gray']
        roi_comp = f['comp']
        # Apply Trinity preprocessing
        roi_tex = apply_clahe(roi_gray)
        roi_geo = get_silhouette(roi_tex)
        roi_gra = get_gradient_map(roi_tex)
        
        # Central Crop for matching
        c_tex = roi_tex[13:43, 13:43]
        c_geo = roi_geo[13:43, 13:43]
        c_gra = roi_gra[13:43, 13:43]
        if is_banner:
            c_tex, c_geo, c_gra = c_tex[12:, :], c_geo[12:, :], c_gra[12:, :]
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            # Check complexity affinity first
            ratio = roi_comp / max(1.0, res['ores'][tier]['mean_comp'])
            affinity = np.exp(-0.5 * (np.log(ratio) / AFFINITY_SIGMA)**2)
            if affinity < 0.1: continue # Early exit for physically impossible matches
            
            best_t_score = 0
            for tpl in res['ores'][tier]['tpls']:
                t_tex, t_geo, t_gra = tpl['tex'][13:43, 13:43], tpl['geo'][13:43, 13:43], tpl['gra'][13:43, 13:43]
                if is_banner:
                    t_tex, t_geo, t_gra = t_tex[12:, :], t_geo[12:, :], t_gra[12:, :]
                
                s_tex = cv2.minMaxLoc(cv2.matchTemplate(t_tex, c_tex, cv2.TM_CCOEFF_NORMED))[1]
                s_geo = cv2.minMaxLoc(cv2.matchTemplate(t_geo, c_geo, cv2.TM_CCOEFF_NORMED))[1]
                s_gra = cv2.minMaxLoc(cv2.matchTemplate(t_gra, c_gra, cv2.TM_CCOEFF_NORMED))[1]
                
                fused = (s_tex * W_TEX) + (s_geo * W_GEO) + (s_gra * W_GRA)
                if fused > best_t_score: best_t_score = fused
            
            frame_results.append({'tier': tier, 'score': best_t_score * affinity})

        if not frame_results: continue
        winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
        # Statistical weighting
        tier_momentum[winner['tier']] += winner['score']
        if winner['score'] > best_overall_score: best_overall_score = winner['score']

    if not tier_momentum: return "low_conf", 0.0, 0, peak_p_score, ""
    winner = max(tier_momentum, key=tier_momentum.get)
    is_valid = best_overall_score >= MIN_FUSED_CONFIDENCE or tier_momentum[winner] > 4.5
    
    if is_valid:
        return winner, round(best_overall_score, 4), int(tier_momentum[winner]), peak_p_score, "[T]" # [T] Trinity

    if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
        val, is_empty = check_side_slice_empty(last_roi_gray, res['bg'], is_banner)
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
                tier, score, harv, pmax, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_score"], results[f"{key}_harv"], results[f"{key}_pmax"], results[f"{key}_tag"] = tier, score, harv, pmax, tag
    return results

def run_tier_identification():
    print(f"--- STEP 4.2: TIER IDENTIFICATION v2.9 (The Trinity Engine) ---")
    if not os.path.exists(BOUNDARIES_CSV) or not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: Missing Input Files.")
        return

    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    buffer_dir, res = cfg.get_buffer_path(0), load_resources()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Trinity: {tag_counts['[T]']}, Likely: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

    final_df = pd.DataFrame(inventory).sort_values('floor_id')
    final_df.to_csv(OUT_CSV, index=False)
    
    print("\nGenerating visual audit proofs...")
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                color = (0, 255, 0) if tier not in ["low_conf", "likely_empty"] else (0, 255, 255) if tier == "likely_empty" else (0, 0, 255)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Check {VERIFY_DIR} for proof images.")

if __name__ == "__main__":
    run_tier_identification()