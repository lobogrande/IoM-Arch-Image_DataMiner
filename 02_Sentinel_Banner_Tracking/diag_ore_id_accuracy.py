# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Tier-Biased Dual-ROI Logic.
# Version: 5.1 (Row-Level Tier Consensus & Family Elections)

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
DIM_OCC = 30  # Surgical ROI for Background/Occupancy (CCOEFF)
DIM_ID  = 48  # Contextual ROI for Tier Identification (Masked CCORR)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# THRESHOLDS
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE    = 0.80  
TIER_CONF_BUFFER   = 0.05  # Buffer for tier-biased conservation (v5.0)

# TIER RANKING (Higher value = Higher complexity/rarity)
TIER_RANK = {
    'dirt1': 1, 'dirt2': 1, 'dirt3': 1,
    'com1': 2, 'com2': 2, 'com3': 2,
    'rare1': 3, 'rare2': 3, 'rare3': 3,
    'epic1': 4, 'epic2': 4, 'epic3': 4,
    'myth1': 5, 'myth2': 5, 'myth3': 5,
    'leg1': 6, 'leg2': 6, 'leg3': 6,
    'div1': 7, 'div2': 7, 'div3': 7
}

def get_family(tier_name):
    """Extracts family name from tier (e.g., 'dirt1' -> 'dirt')."""
    return ''.join([i for i in tier_name if not i.isdigit()])

def get_spatial_mask():
    mask = np.zeros((DIM_ID, DIM_ID), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore_occ': {}, 'ore_id': {}, 'bg_occ': [], 'bg_id': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        h, w = img_raw.shape
        cy, cx = h // 2, w // 2
        r = DIM_OCC // 2
        img_occ = img_raw[cy-r : cy+r, cx-r : cx+r]
        img_id = cv2.resize(img_raw, (DIM_ID, DIM_ID))
            
        if f.startswith("background") or f.startswith("negative_ui"):
            templates['bg_occ'].append({'id': f, 'img': img_occ})
            templates['bg_id'].append({'id': f, 'img': img_id})
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['ore_occ']:
                templates['ore_occ'][tier] = {'act': [], 'sha': []}
                templates['ore_id'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']:
                templates['ore_occ'][tier][state].append({'id': f, 'img': img_occ})
                templates['ore_id'][tier][state].append({'id': f, 'img': img_id})
    return templates

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir):
    """Worker function using Row-Level Consensus logic."""
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return [], []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    
    # --- PHASE 1: Collect ALL Potential Matches for the Row ---
    slot_matches = {} # col -> list of all tier best-matches

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        
        # DNA Gate
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna'}
            continue

        # 30px Occupancy Gate
        x1_occ, y1_occ = int(cx - (DIM_OCC//2) - JITTER), int(row4_y - (DIM_OCC//2) - JITTER)
        search_occ = img_gray[y1_occ : y1_occ + DIM_OCC + (JITTER*2), x1_occ : x1_occ + DIM_OCC + (JITTER*2)]
        best_bg_occ = 0.0
        for bg_tpl in templates['bg_occ']:
            res = cv2.matchTemplate(search_occ, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            best_bg_occ = max(best_bg_occ, cv2.minMaxLoc(res)[1])

        if best_bg_occ > BG_OCCUPANCY_FLOOR:
            slot_matches[col] = {'status': 'empty_bg', 'occ_score': best_bg_occ}
            continue

        # 48px Masked Identity phase - Run ALL tiers to prepare for election
        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        
        tier_performances = {} # tier -> best score in this slot
        for tier, states in templates['ore_id'].items():
            best_score = -1.0
            best_id = 'none'
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_id, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    score = cv2.minMaxLoc(res)[1]
                    if score > best_score:
                        best_score = score
                        best_id = ore_tpl['id']
            tier_performances[tier] = {'score': best_score, 'id': best_id}
            
        slot_matches[col] = {'status': 'occupied', 'occ_score': best_bg_occ, 'tiers': tier_performances}

    # --- PHASE 2: ROW-LEVEL TIER ELECTION ---
    # Determine which specific tier of a family is dominant in this row.
    family_champions = {} # family -> {'tier': 'dirt1', 'score': 0.94}
    
    for col, data in slot_matches.items():
        if data['status'] != 'occupied': continue
        for tier, perf in data['tiers'].items():
            fam = get_family(tier)
            if fam not in family_champions or perf['score'] > family_champions[fam]['score']:
                family_champions[fam] = {'tier': tier, 'score': perf['score']}

    # --- PHASE 3: FINAL RESOLUTION ---
    frame_results = []
    has_detections = False

    for col in range(6):
        data = slot_matches[col]
        if data['status'] == 'empty_dna':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_dna', 'occ_score': 0.0, 'id_score': 0.0, 'margin': 0.0, 'ore_id': 'none', 'was_corrected': False})
            continue
        if data['status'] == 'empty_bg':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_bg', 'occ_score': round(data['occ_score'], 4), 'id_score': 0.0, 'margin': 0.0, 'ore_id': 'none', 'was_corrected': False})
            continue

        # Get initial local winner using v5.0 Tier Bias
        tier_list = [{'tier': t, 'score': p['score'], 'id': p['id']} for t, p in data['tiers'].items()]
        tier_list.sort(key=lambda x: x['score'], reverse=True)
        
        local_winner = tier_list[0]
        # v5.0 Downgrade check
        for challenger in tier_list[1:]:
            if TIER_RANK.get(challenger['tier'], 0) < TIER_RANK.get(local_winner['tier'], 0):
                if (local_winner['score'] - challenger['score']) < TIER_CONF_BUFFER:
                    local_winner = challenger

        # v5.1 Row-Level Consensus Check
        final_winner = local_winner
        was_corrected = False
        
        fam = get_family(local_winner['tier'])
        champion = family_champions.get(fam)
        
        # If the row has a 'champion' tier for this family that is different from our local winner
        if champion and champion['tier'] != local_winner['tier']:
            # And the champion tier actually matched reasonably well here (e.g. > 0.70)
            champ_perf = data['tiers'].get(champion['tier'])
            if champ_perf and champ_perf['score'] > 0.70:
                final_winner = {'tier': champion['tier'], 'score': champ_perf['score'], 'id': champ_perf['id']}
                was_corrected = True

        is_valid = final_winner['score'] > ORE_STRICT_GATE
        detected = final_winner['tier'] if is_valid else "low_conf_id"
        
        # Visual Annotation
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        if was_corrected: color = (255, 0, 255) # Purple for consensus corrections
        
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        cv2.putText(img_color, f"{detected} ({final_winner['score']:.2f})", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if is_valid: has_detections = True

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'occ_score': round(data['occ_score'], 4), 'id_score': round(final_winner['score'], 4),
            'ore_id': final_winner['id'], 'was_corrected': was_corrected
        })

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"consensus_v51_f{f_idx}.jpg"), img_color)

    return frame_results, []

def run_ore_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v5.1: ROW-LEVEL CONSENSUS ---")

    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res, _ = future.result()
            all_results.extend(res)
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v5.1_consensus.csv"), index=False)
    
    print(f"\n--- CONSENSUS STATS ---")
    corrections = audit_df['was_corrected'].sum()
    print(f"Total Ores Forced to Champion Tier for Consistency: {corrections}")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_ore_audit()