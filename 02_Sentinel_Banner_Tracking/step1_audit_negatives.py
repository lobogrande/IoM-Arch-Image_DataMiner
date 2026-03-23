# step1_audit_negatives.py
# Purpose: Forensic analysis of all frames excluded by Step 1 to detect false negatives.
# Version: 3.1 (Empty File Recovery)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_negative_audit():
    # --- 1. RESOLVE DATA PATHS ---
    source_dir = cfg.get_buffer_path() 
    run_id = os.path.basename(source_dir).split('_')[-1]
    
    csv_path = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{run_id}.csv")
    audit_out_path = os.path.join(cfg.DATA_DIRS["TRACKING"], f"audit_negatives_run_{run_id}.csv")
    false_neg_path = os.path.join(cfg.DATA_DIRS["TRACKING"], f"check_false_negatives_run_{run_id}.csv")

    print(f"--- STEP 1 AUDIT: NEGATIVE SPACE (Target: Run {run_id}) ---")
    
    # 2. IDENTIFY DISCARDED FRAMES (Handled Empty Files)
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    hit_files = set()

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            hit_df = pd.read_csv(csv_path)
            if not hit_df.empty:
                hit_files = set(hit_df['filename'].tolist())
        except pd.errors.EmptyDataError:
            print(f" [!] Note: {os.path.basename(csv_path)} contains no data.")
    else:
        print(f" [!] Note: {os.path.basename(csv_path)} is missing or empty. Scanning full buffer.")

    discard_files = [f for f in all_files if f not in hit_files]
    
    print(f" Total Frames in Buffer: {len(all_files)}")
    print(f" Current Master Hits:    {len(hit_files)}")
    print(f" Discard Pool to Scan:   {len(discard_files)}")

    if not discard_files:
        print(" [SUCCESS] No frames left to audit.")
        return

    # 3. GLOBAL SEARCH (Scanning Rows 1-4 Band)
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    audit_results = []
    # Band covering Rows 1-4
    y_min, y_max = 200, 500 

    print(f"Profiling discards for peak signal distribution...")
    for f_idx, filename in enumerate(discard_files):
        img = cv2.imread(os.path.join(source_dir, filename), 0)
        if img is None: continue
        
        strip = img[y_min:y_max, :]
        res_r = cv2.matchTemplate(strip, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
        res_l = cv2.matchTemplate(strip, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
        
        is_right = val_r >= val_l
        best_val = val_r if is_right else val_l
        best_y = (loc_r[1] if is_right else loc_l[1]) + y_min
        best_x = loc_r[0] if is_right else loc_l[0]

        audit_results.append({
            'filename': filename,
            'peak_x': best_x,
            'peak_y': best_y,
            'confidence': round(best_val, 4),
            'direction': "Right" if is_right else "Left"
        })

        if f_idx % 2000 == 0:
            print(f"  Processed {f_idx}/{len(discard_files)} discards...")

    # 4. ANALYSIS & CATEGORIZATION
    df = pd.DataFrame(audit_results)
    
    # Row assignment
    df['row'] = 0
    df.loc[df['peak_y'] < 290, 'row'] = 1
    df.loc[(df['peak_y'] >= 290) & (df['peak_y'] < 349), 'row'] = 2
    df.loc[(df['peak_y'] >= 349) & (df['peak_y'] < 408), 'row'] = 3
    df.loc[df['peak_y'] >= 408, 'row'] = 4

    print("\n--- DISCARD DISTRIBUTION SUMMARY ---")
    if not df.empty:
        summary = df.groupby('row').agg({'filename':'count', 'confidence':'mean'}).rename(columns={'filename':'count', 'confidence':'avg_conf'})
        print(summary)
        df.to_csv(audit_out_path, index=False)
        
        # Identify False Negatives
        false_neg_candidates = df[(df['row'].isin([1, 2])) & (df['confidence'] > 0.70)]
        if not false_neg_candidates.empty:
            print(f"\n[!] ALERT: {len(false_neg_candidates)} High-Confidence misses found in Rows 1/2.")
            print(f"Check: {os.path.basename(false_neg_path)}")
            false_neg_candidates.to_csv(false_neg_path, index=False)
        else:
            print("\n[SUCCESS] No significant false negatives detected.")
    else:
        print(" No data to analyze.")

if __name__ == "__main__":
    run_negative_audit()