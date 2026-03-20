# sprite_negative_audit.py
# Purpose: Forensic analysis of all frames excluded by the Master Sequencer.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_negative_audit():
    print("--- NEGATIVE SPACE AUDIT: COMMENCING ---")
    
    # 1. Identify Discards
    source_dir = cfg.get_buffer_path(0)
    csv_path = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
    
    if not os.path.exists(csv_path):
        print("Error: Master CSV not found. Run v2.8 first.")
        return
        
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    hit_df = pd.read_csv(csv_path)
    hit_files = set(hit_df['filename'].tolist())
    discard_files = [f for f in all_files if f not in hit_files]
    
    print(f"Total Frames:   {len(all_files)}")
    print(f"Master Hits:    {len(hit_files)}")
    print(f"Discard Pool:   {len(discard_files)}")

    # 2. Global Player Locator (Scanning rows 1-4)
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    audit_results = []
    # Narrow Y-strip for efficiency (Covers Rows 1, 2, 3, 4)
    y_min, y_max = 200, 500 

    print(f"Profiling discards for Y-coordinate distribution...")
    for f_idx, filename in enumerate(discard_files):
        img = cv2.imread(os.path.join(source_dir, filename), 0)
        if img is None: continue
        
        # Scan entire width in the grid Y-range
        strip = img[y_min:y_max, :]
        
        # Check both directions
        res_r = cv2.matchTemplate(strip, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
        
        res_l = cv2.matchTemplate(strip, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
        
        # Record the better match
        best_val = max(val_r, val_l)
        best_y = (loc_r[1] if val_r >= val_l else loc_l[1]) + y_min
        best_x = loc_r[0] if val_r >= val_l else loc_l[0]
        direction = "Right" if val_r >= val_l else "Left"

        audit_results.append({
            'filename': filename,
            'peak_x': best_x,
            'peak_y': best_y,
            'confidence': round(best_val, 4),
            'direction': direction
        })

        if f_idx % 2000 == 0:
            print(f"  Processed {f_idx}/{len(discard_files)} discards...")

    # 3. Categorization & Summary
    df = pd.DataFrame(audit_results)
    
    # Define Row Boundaries based on 59px step
    # Row 1 ~225, Row 2 ~284, Row 3 ~343, Row 4 ~402
    df['row'] = 0
    df.loc[df['peak_y'] < 255, 'row'] = 1
    df.loc[(df['peak_y'] >= 255) & (df['peak_y'] < 314), 'row'] = 2
    df.loc[(df['peak_y'] >= 314) & (df['peak_y'] < 373), 'row'] = 3
    df.loc[df['peak_y'] >= 373, 'row'] = 4

    print("\n--- AUDIT DISCARD DISTRIBUTION ---")
    summary = df.groupby('row').agg({'filename':'count', 'confidence':'mean'}).rename(columns={'filename':'count', 'confidence':'avg_conf'})
    print(summary)

    # Save for manual verification of "High Confidence Discards"
    df.to_csv("discard_audit_full.csv", index=False)
    
    # Isolate Potential False Negatives (Row 1/2 with high confidence but excluded)
    false_neg_candidates = df[(df['row'].isin([1, 2])) & (df['confidence'] > 0.70)]
    print(f"\nPotential False Negatives in Rows 1/2: {len(false_neg_candidates)}")
    if not false_neg_candidates.empty:
        print("Saving candidates to 'false_negatives_check.csv'...")
        false_neg_candidates.to_csv("false_negatives_check.csv", index=False)

if __name__ == "__main__":
    run_negative_audit()