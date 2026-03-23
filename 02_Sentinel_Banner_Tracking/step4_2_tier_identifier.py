# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using the Forensic Trinity:
#          Triple-Sensor Fusion, Context-Aware Forensics, and Structural Filtering.
# Version: 3.5 (The Context-Aware Auditor)

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

# --- 2. DATA-DRIVEN CONSTANTS (Calibrated) ---
SIDE_SLICE_WIDTH = 12        
SIDE_SLICE_STD_MAX = 11.0    
Z_TRUST_THRESHOLD = 1.2      # Softened for high-noise buffer
MIN_MOMENTUM_GATE = 2.5      # Lowered to recover quiet consensus
PLAYER_PRESENCE_TRESHOLD = 0.45 # Trigger for likely_empty forensics
COMPLEXITY_DIRT_CEILING = 480.0
COMPLEXITY_HIGH_FLOOR = 720.0

# Trinity Weights: Favor Grayscale (Tex) for signal stability in noise
W_TEX, W_GEO, W_GRA = 0.50, 0.25, 0.25
HARVEST_COUNT = 15          

def get_complexity(img):
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def get_silhouette(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh[SIDE_PX//2, SIDE_PX//2] == 0: thresh = cv2.bitwise_not(thresh)
    return thresh

def get_gradient_map(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_resources():
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
            img_tex = apply_clahe(img_scaled)
            res['ores'][tier]['tpls'].append({
                'tex': img_tex, 'geo': get_silhouette(img_tex), 'gra': get_gradient_map(img_tex),
                'comp': get_complexity(img_scaled)
            })
        if "negative_player" in f: res['player'].append(img_scaled)
        if "background_plain" in f: res['bg'].append(img_scaled)
    for tier in res['ores']:
        res['ores'][tier]['mean_comp'] = np.mean([t['comp'] for t in res['ores'][tier]['tpls']])
    return res

def check_side_slice_empty(roi_gray):
    """Forensic Fallback: Peeks at side-slice for background smoothness."""
    slot_48 = roi_gray[4:52, 4:52]
    slice_roi = slot_48[:, 0:SIDE_SLICE_WIDTH]
    std_val = np.std(slice_roi)
    return std_val, std_val < SIDE_SLICE_STD_MAX

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Consensus engine with Context-Aware forensic fallback."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score, best_roi_gray = 0.0, None

    # 1. Harvest frames
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_gray = cv2.cvtColor(img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX], cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        comp = get_complexity(roi_gray)
        if best_roi_gray is None or comp > get_complexity(best_roi_gray):
            best_roi_gray = roi_gray
            
        if max_p < 0.75 and comp > 200:
            frame_candidates.append({'gray': roi_gray, 'comp': comp})

    # 2. Identification Logic
    tier_z_momentum = defaultdict(float)
    max_z_seen = 0.0
    top_frames = sorted(frame_candidates, key=lambda x: x['comp'], reverse=True)[:HARVEST_COUNT]
    
    for f in top_frames:
        roi_gray, roi_comp = f['gray'], f['comp']
        roi_tex = apply_clahe(roi_gray)
        roi_geo, roi_gra = get_silhouette(roi_tex), get_gradient_map(roi_tex)
        c_tex, c_geo, c_gra = roi_tex[13:43, 13:43], roi_geo[13:43, 13:43], roi_gra[13:43, 13:43]
        if is_banner: c_tex, c_geo, c_gra = c_tex[12:, :], c_geo[12:, :], c_gra[12:, :]
        
        # Band-Pass candidates
        valid_pool = allowed_tiers
        if roi_comp > COMPLEXITY_HIGH_FLOOR:
            valid_pool = [t for t in allowed_tiers if not any(k in t for k in ['dirt', 'com'])]
        elif roi_comp < COMPLEXITY_DIRT_CEILING:
            valid_pool = [t for t in allowed_tiers if any(k in t for k in ['dirt', 'com'])]

        frame_results = []
        for tier in valid_pool:
            if tier not in res['ores']: continue
            best_t_score = 0
            for tpl in res['ores'][tier]['tpls']:
                t_tex, t_geo, t_gra = tpl['tex'][13:43, 13:43], tpl['geo'][13:43, 13:43], tpl['gra'][13:43, 13:43]
                if is_banner: t_tex, t_geo, t_gra = t_tex[12:, :], t_geo[12:, :], t_gra[12:, :]
                
                s_tex = cv2.minMaxLoc(cv2.matchTemplate(t_tex, c_tex, cv2.TM_CCOEFF_NORMED))[1]
                s_geo = cv2.minMaxLoc(cv2.matchTemplate(t_geo, c_geo, cv2.TM_CCOEFF_NORMED))[1]
                s_gra = cv2.minMaxLoc(cv2.matchTemplate(t_gra, c_gra, cv2.TM_CCOEFF_NORMED))[1]
                
                fused = (s_tex * W_TEX) + (s_geo * W_GEO) + (s_gra * W_GRA)
                if fused > best_t_score: best_t_score = fused
            frame_results.append({'tier': tier, 'score': best_t_score})

        if not frame_results: continue
        scores = np.array([x['score'] for x in frame_results])
        if len(scores) > 1:
            mean_s, std_s = np.mean(scores), np.std(scores)
            winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
            z_score = (winner['score'] - mean_s) / max(0.01, std_s)
            if z_score > 0.8: # Minimal signal requirement
                tier_z_momentum[winner['tier']] += z_score
                max_z_seen = max(max_z_seen, z_score)

    # 3. Resolution Hierarchy
    if tier_z_momentum:
        winner_tier = max(tier_z_momentum, key=tier_z_momentum.get)
        if tier_z_momentum[winner_tier] >= MIN_MOMENTUM_GATE or max_z_seen >= Z_TRUST_THRESHOLD:
            tag = "[Z]" if max_z_seen >= Z_TRUST_THRESHOLD else "[M]"
            return winner_tier, round(max_z_seen, 2), int(tier_z_momentum[winner_tier]), peak_p_score, tag

    # 4. Forensic Fallback (ONLY if player presence was detected)
    if peak_p_score > PLAYER_PRESENCE_TRESHOLD and best_roi_gray is not None:
        val, is_empty = check_side_slice_empty(best_roi_gray)
        if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    return "low_conf", round(max_z_seen, 2), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    if hasattr(cfg, 'BOSS_DATA') and f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            key = f"R{r+1}_S{c}"
            results[key], results[f"{key}_tag"] = identity, "[B]"
            results[f"{key}_z"], results[f"{key}_mom"], results[f"{key}_pmax"] = 0.0, 0, 0.0
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1)
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key], results[f"{key}_tag"] = "empty", ""
                results[f"{key}_z"], results[f"{key}_mom"], results[f"{key}_pmax"] = 0.0, 0, 0.0
            else:
                tier, z_score, momentum, pmax, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_z"], results[f"{key}_mom"], results[f"{key}_pmax"], results[f"{key}_tag"] = tier, z_score, momentum, pmax, tag
    return results

def run_tier_identification():
    print(f"--- STEP 4.2: TIER IDENTIFICATION v3.5 (Context-Aware) ---")
    if not os.path.exists(DNA_INVENTORY_CSV): return
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
            print(f"  Floor {f_id:03d} processed. [Success: {tag_counts['[Z]']}+{tag_counts['[M]']}, LikelyEmpty: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    
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
                
                z_val = row.get(f"{key}_z", 0.0)
                mom_val = row.get(f"{key}_mom", 0)
                if not pd.isna(z_val) and not pd.isna(mom_val) and tier not in ["likely_empty", "low_conf"]:
                    cv2.putText(img, f"Z:{z_val:.1f} M:{int(mom_val)}", (cx+HUD_DX-25, cy+HUD_DY+12), 0, 0.3, (255,255,255), 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()