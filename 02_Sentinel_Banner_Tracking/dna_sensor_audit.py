# dna_sensor_audit.py
# Purpose: Profile and verify background template matching for Row 3/4 occupancy detection.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS GRID CONSTANTS (Ore Centers)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# Initial threshold for profiling - will be refined by the output stats.
BG_MATCH_THRESHOLD = 0.85

def load_all_bg_templates():
    """Loads all available background and negative UI templates."""
    templates = []
    # Load plain backgrounds
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(p):
            templates.append({'id': f'plain_{i}', 'img': cv2.imread(p, 0)})
    
    # Load negative UI (often contains banner artifacts)
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"negative_ui_{i}.png")
        if os.path.exists(p):
            templates.append({'id': f'neg_ui_{i}', 'img': cv2.imread(p, 0)})
            
    return templates

def get_slot_profile(img, row_idx, col_idx, templates):
    """Matches a specific slot against all templates and returns the best score."""
    y_center = int(ORE0_Y + (row_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    
    tw, th = 30, 30
    tx, ty = x_center - (tw // 2), y_center - (th // 2)
    roi = img[ty : ty + th, tx : tx + tw]
    
    if roi.shape[0] < th or roi.shape[1] < tw:
        return 0, "none"
        
    best_val = -1
    best_id = "none"
    
    for t in templates:
        res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_id = t['id']
            
    return best_val, best_id

def run_dna_profiling_audit():
    input_csv = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    if not os.path.exists(input_csv):
        print("Error: sprite_homing_run_0.csv not found.")
        return

    bg_templates = load_all_bg_templates()
    if not bg_templates:
        print("Error: No background templates found.")
        return

    df = pd.read_csv(input_csv)
    source_dir = cfg.get_buffer_path(0)
    
    print(f"--- DNA SENSOR PROFILING (Templates: {len(bg_templates)}) ---")
    
    profile_data = []
    results = []
    
    if not os.path.exists("dna_debug"): os.makedirs("dna_debug")

    # We'll profile the first 1000 frames to get a solid statistical baseline
    sample_limit = min(len(df), 1000)

    for idx, row in df.iterrows():
        if idx >= sample_limit: break
        
        img = cv2.imread(os.path.join(source_dir, row['filename']), 0)
        if img is None: continue
        
        r3_bits = []
        r4_bits = []
        
        # Profile Row 3 and Row 4
        for r_idx in [2, 3]:
            for c_idx in range(6):
                score, t_id = get_slot_profile(img, r_idx, c_idx, bg_templates)
                
                profile_data.append({
                    'frame': row['frame_idx'],
                    'row': r_idx + 1,
                    'col': c_idx,
                    'max_conf': score,
                    'best_tpl': t_id
                })
                
                bit = '0' if score >= BG_MATCH_THRESHOLD else '1'
                if r_idx == 2: r3_bits.append(bit)
                else: r4_bits.append(bit)

        r3_dna = "".join(r3_bits)
        r4_dna = "".join(r4_bits)
        combined = f"{r3_dna}-{r4_dna}"

        # Visual Verification for every 100th frame
        if idx % 100 == 0:
            vis = cv2.imread(os.path.join(source_dir, row['filename']))
            for r_idx in [2, 3]:
                y = int(ORE0_Y + (r_idx * STEP))
                current_dna = r3_dna if r_idx == 2 else r4_dna
                for c_idx in range(6):
                    x = int(ORE0_X + (c_idx * STEP))
                    color = (0, 255, 0) if current_dna[c_idx] == '1' else (0, 0, 255)
                    cv2.circle(vis, (x, y), 6, color, -1)
            cv2.imwrite(f"dna_debug/dna_prof_f{row['frame_idx']}.jpg", vis)

        results.append({'frame_idx': row['frame_idx'], 'dna_sig': combined})

    # 1. Distribution Analysis
    prof_df = pd.DataFrame(profile_data)
    prof_df.to_csv("dna_profiling_stats.csv", index=False)
    
    print("\n--- MATCH SCORE DISTRIBUTION (EMPTY CANDIDATES) ---")
    summary = prof_df.groupby(['row', 'col'])['max_conf'].describe(percentiles=[.01, .05, .5, .95, .99])
    print(summary[['min', '5%', '50%', 'max']])
    
    # 2. Signature distribution
    audit_df = pd.DataFrame(results)
    print("\n--- DNA SIGNATURES FOUND (Using 0.85 Threshold) ---")
    print(audit_df['dna_sig'].value_counts().head(5))
    
    print(f"\n[DONE] Check 'dna_profiling_stats.csv' to determine the optimal threshold.")

if __name__ == "__main__":
    run_dna_profiling_audit()