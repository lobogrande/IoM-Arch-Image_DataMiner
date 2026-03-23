# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Temporal Consensus,
#          Structural Affinity Weighting, and Variance-Based Side-Slice Forensics.
# Version: 2.5 (The Structural Affinity Engine)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. UNABRIDGED BOSS DATA (Project Mandate) ---
BOSS_DATA = {
    11:  {'tier': 'dirt1'},
    22:  {'tier': 'dirt2'},
    33:  {'tier': 'com1'},
    44:  {'tier': 'com2'},
    55:  {'tier': 'rare1'},
    66:  {'tier': 'rare2'},
    77:  {'tier': 'epic1'},
    88:  {'tier': 'epic2'},
    98:  {'tier': 'mixed', 'special': {
            0: 'dirt1', 1: 'dirt2', 2: 'com1', 3: 'com2', 4: 'rare1', 5: 'rare2',
            6: 'epic1', 7: 'epic2', 8: 'leg1', 9: 'leg2', 10: 'myth1', 11: 'myth2',
            12: 'dirt1', 13: 'dirt2', 14: 'com1', 15: 'com2', 16: 'rare1', 17: 'rare2',
            18: 'epic1', 19: 'epic2', 20: 'leg1', 21: 'leg2', 22: 'myth1', 23: 'myth2'
         }},
    99:  {'tier': 'mixed', 'special': {
            0: 'dirt1', 1: 'dirt2', 2: 'com1', 3: 'com2', 4: 'rare1', 5: 'rare2',
            6: 'epic1', 7: 'epic2', 8: 'leg1', 9: 'leg2', 10: 'myth1', 11: 'myth2',
            12: 'dirt1', 13: 'dirt2', 14: 'com1', 15: 'com2', 16: 'rare1', 17: 'rare2',
            18: 'epic1', 19: 'epic2', 20: 'leg1', 21: 'leg2', 22: 'myth1', 23: 'myth2'
         }},
    100: {'tier': 'myth1'},
    110: {'tier': 'myth2'}
}

# --- 2. GRID & HUD CONSTANTS (Verified) ---
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

# MATHEMATICAL GATES
MIN_MATCH_CONFIDENCE = 0.44  
STRUCTURAL_WEIGHT = 0.15     # Penalty multiplier for complexity mismatch
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_WIDTH = 14        # Peeking further past player torso
SIDE_SLICE_STD_MAX = 14.0    # Background smoothness threshold
HARVEST_COUNT = 15          
COMPLEXITY_GATE = 320        

def get_complexity(img):
    """Calculates Laplacian variance as a measure of structural signal."""
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def load_resources():
    """Loads ore templates and profiles their structural complexity."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    print("Profiling Template Library...")
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_48 = cv2.resize(img, (48, 48))
        
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            # Store image and its inherent complexity
            res['ores'][tier].append({
                'img': img_48,
                'comp': get_complexity(img_48)
            })
        if "negative_player" in f: res['player'].append(img_48)
        if "background_plain" in f: res['bg'].append(img_48)
            
    return res

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """Forensic check: Peeks at the side-slice for 'Pure Background' smoothness."""
    slot_48 = roi_gray[4:52, 4:52]
    if is_banner: slot_48 = slot_48[12:, :]
    
    slice_roi = slot_48[:, 0:SIDE_SLICE_WIDTH]
    
    # 1. Variance Check (Smoothness)
    std_dev = np.std(slice_roi)
    if std_dev < SIDE_SLICE_STD_MAX:
        return std_dev, True

    # 2. Template Correlation Check
    best_s = 0
    for tpl in bg_tpls:
        tpl_slice = tpl[:, 0:SIDE_SLICE_WIDTH]
        if is_banner: tpl_slice = tpl_slice[12:, :]
        res = cv2.matchTemplate(tpl_slice, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
        
    return best_s, best_s > 0.70

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Consensus engine driven by Structural Affinity (Complexity Delta)."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score = 0.0
    last_roi_gray = None

    # 1. HARVEST
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
            comp = get_complexity(roi_gray)
            if comp >= COMPLEXITY_GATE:
                frame_candidates.append({'gray': roi_gray, 'comp': comp})

    # 2. MATCH WITH STRUCTURAL AFFINITY
    tier_tallies, valid_matches = [], []
    best_overall_score = 0
    
    top_frames = sorted(frame_candidates, key=lambda x: x['comp'], reverse=True)[:HARVEST_COUNT]
    
    for f in top_frames:
        roi_30 = f['gray'][13:43, 13:43]
        if is_banner: roi_30 = roi_30[12:, :]
        roi_comp = f['comp']
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            for tpl_data in res['ores'][tier]:
                # Standard Correlation
                score = cv2.minMaxLoc(cv2.matchTemplate(tpl_data['img'], roi_30, cv2.TM_CCOEFF_NORMED))[1]
                
                # STRUCTURAL AFFINITY PENALTY
                # Calculate how different the template is from the ROI in terms of detail.
                # Normalized by a log scale to handle high-variance complexity.
                comp_delta = abs(np.log1p(roi_comp) - np.log1p(tpl_data['comp']))
                affinity_penalty = comp_delta * STRUCTURAL_WEIGHT
                
                final_score = score - affinity_penalty
                frame_results.append({'tier': tier, 'score': final_score})

        if not frame_results: continue
        
        # Determine unique winner for this frame
        winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
        tier_tallies.append(winner['tier'])
        
        if winner['score'] > best_overall_score: best_overall_score = winner['score']
        if winner['score'] >= MIN_MATCH_CONFIDENCE:
            valid_matches.append(winner['tier'])

    # 3. RESOLUTION
    if valid_matches:
        vote_winner, vote_count = Counter(valid_matches).most_common(1)[0]
        return vote_winner, round(best_overall_score, 4), vote_count, peak_p_score, "[A]" # [A] Affinity Match

    if tier_tallies:
        vote_winner, vote_count = Counter(tier_tallies).most_common(1)[0]
        if best_overall_score >= 0.38:
            return vote_winner, round(best_overall_score, 4), vote_count, peak_p_score, "[P]"

    # 4. SIDE-SLICE FORENSICS
    if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
        val, is_empty = check_side_slice_empty(last_roi_gray, res['bg'], is_banner)
        if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    return "low_conf", round(best_overall_score, 4), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    if f_id in BOSS_DATA:
        boss = BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_tag"] = ""
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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v2.5 (Structural Affinity) ---")
    
    if not os.path.exists(BOUNDARIES_CSV) or not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: Required CSV files missing.")
        return

    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    buffer_dir, res = cfg.get_buffer_path(0), load_resources()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"Parallelizing {len(df_floors)} floors with Structural Affinity Weighting...")
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Affinity: {tag_counts['[A]']}, LikelyEmpty: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

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
    print(f"[DONE] Audit complete. Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()