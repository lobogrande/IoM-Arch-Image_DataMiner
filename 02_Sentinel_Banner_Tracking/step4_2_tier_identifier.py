# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Temporal Consensus,
#          Variance-Based Side-Slice Forensics, and Bully Penalties.
# Version: 2.3 (The Forensic Stabilizer)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- GRID & HUD CONSTANTS (Verified) ---
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

# THRESHOLDS
MIN_MATCH_CONFIDENCE = 0.45  
PROMOTION_THRESHOLD = 0.38   
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_GATE = 0.65       
HARVEST_COUNT = 15          
COMPLEXITY_GATE = 350        

# TIER PENALTIES: Prevent generic textures from "Bullying" specific features
BULLY_PENALTIES = {
    'dirt1': 0.05, 'dirt2': 0.06, 'dirt3': 0.04,
    'com1':  0.03, 'com2':  0.03, 'com3':  0.02
}

def load_resources():
    """Loads ore tiers, player sprites, and background templates."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_48 = cv2.resize(img, (48, 48))
        
        # 1. Pure Ore Tiers
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            res['ores'][tier].append(img_48)
            
        # 2. Player Sprites (Obstruction Check)
        if "negative_player" in f: 
            res['player'].append(img_48)
            
        # 3. Background Templates (Forensic Peek Only)
        if "background_plain" in f: 
            res['bg'].append(img_48)
            
    return res

def apply_gamma_lift(img, gamma=0.6):
    """Normalizes lighting to reveal latent structural features in dark ores."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """
    Forensic check: Peeks at the left 10px strip for background presence.
    Uses both Correlation and Variance (Background is smooth, Ores are textured).
    """
    slot_48 = roi_gray[4:52, 4:52]
    if is_banner: slot_48 = slot_48[12:, :]
    
    slice_roi = slot_48[:, 0:10]
    
    # Variance check: Background is relatively uniform
    std_dev = np.std(slice_roi)
    if std_dev < 12.0: # Mathematically "Smooth"
        return std_dev, True

    # Template correlation fallback
    best_s = 0
    for tpl in bg_tpls:
        tpl_slice = tpl[:, 0:10]
        if is_banner: tpl_slice = tpl_slice[12:, :]
        res = cv2.matchTemplate(tpl_slice, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
        
    return best_s, best_s > SIDE_SLICE_GATE

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Temporal consensus search with Bully Penalties and Gamma Lifting."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score = 0.0
    last_roi_gray = None

    # 1. HARVEST CLEAN FRAMES
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_gray = cv2.cvtColor(img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX], cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        last_roi_gray = roi_gray
        
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        if max_p < PLAYER_REJECTION_GATE:
            comp = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            if comp >= COMPLEXITY_GATE:
                frame_candidates.append({'gray': roi_gray, 'quality': comp})

    # 2. IDENTIFICATION
    tier_tallies, valid_matches = [], []
    best_overall_score = 0
    
    top_frames = sorted(frame_candidates, key=lambda x: x['quality'], reverse=True)[:HARVEST_COUNT]
    
    for f in top_frames:
        # Normalization: Apply Gamma lift to see into the dark
        roi_lifted = apply_gamma_lift(f['gray'])
        roi_30 = roi_lifted[13:43, 13:43]
        if is_banner: roi_30 = roi_30[12:, :]
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            s = max([cv2.minMaxLoc(cv2.matchTemplate(tpl, roi_30, cv2.TM_CCOEFF_NORMED))[1] for tpl in res['ores'][tier]])
            # Apply Bully Penalty
            s -= BULLY_PENALTIES.get(tier, 0.0)
            frame_results.append({'tier': tier, 'score': s})

        if not frame_results: continue
        
        # Find best for this frame
        winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
        tier_tallies.append(winner['tier'])
        
        if winner['score'] > best_overall_score: best_overall_score = winner['score']
        if winner['score'] >= MIN_MATCH_CONFIDENCE:
            valid_matches.append(winner['tier'])

    # 3. RESOLUTION HIERARCHY
    if valid_matches:
        vote_winner, vote_count = Counter(valid_matches).most_common(1)[0]
        return vote_winner, round(best_overall_score, 4), vote_count, peak_p_score, ""

    if tier_tallies:
        vote_winner, vote_count = Counter(tier_tallies).most_common(1)[0]
        if best_overall_score >= PROMOTION_THRESHOLD:
            return vote_winner, round(best_overall_score, 4), vote_count, peak_p_score, "[P]"

    # 4. FORENSIC FALLBACK (Likely Empty)
    if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
        val, is_empty = check_side_slice_empty(last_roi_gray, res['bg'], is_banner)
        if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    return "low_conf", round(best_overall_score, 4), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    # 1. BOSS DATA ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_tag"] = ""
        return results

    # 2. CONFIG FILTERING
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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v2.3 ---")
    
    if not os.path.exists(BOUNDARIES_CSV) or not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: Required CSV files missing.")
        return

    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    buffer_dir, res = cfg.get_buffer_path(0), load_resources()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"Initialized Resources. Parallelizing {len(df_floors)} floors...")
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Promoted: {tag_counts['[P]']}, LikelyEmpty: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

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
                # Text labels with forensic markers
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Audit complete. Check {VERIFY_DIR}")

if __name__ == "__main__":
    run_tier_identification()