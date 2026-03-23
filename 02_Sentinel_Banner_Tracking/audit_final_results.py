# audit_final_results.py
# Purpose: Verify the logical and mathematical integrity of the 110-floor dataset.
# Version: 1.2 (Corrected Forensic Whitelist)

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_integrity_audit():
    # Resolve path using project config
    csv_path = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
    
    if not os.path.exists(csv_path):
        print(f"!!! ERROR: Could not find inventory at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    errors = []
    
    print(f"--- STARTING FINAL DATASET AUDIT ---")
    print(f"Target File: {csv_path}\n")
    
    for _, row in df.iterrows():
        f_id = int(row['floor_id'])
        
        # 1. Violation Check: ORE_RESTRICTIONS
        for r in range(1, 5):
            for s in range(6):
                slot_key = f"R{r}_S{s}"
                tier = str(row[slot_key])
                if tier in ["empty", "low_conf", "likely_empty", "obstructed", "unknown_obstructed"]: continue
                
                limit = cfg.ORE_RESTRICTIONS.get(tier)
                if limit and not (limit[0] <= f_id <= limit[1]):
                    errors.append(f"Floor {f_id}: {tier} found in {slot_key} (FORBIDDEN by biome restrictions)")

        # 2. Forensic Check: The "Cyan/Yellow" Rule
        for r in range(1, 5):
            for s in range(6):
                tag_col = f"R{r}_S{s}_tag"
                if tag_col in row:
                    tag = str(row[tag_col])
                    if tag in ["[L]", "[U]"]:
                        # CORRECTED: F2 (S1), F3 (S3), F7 (S0)
                        valid_spots = [(2, "R1_S1"), (3, "R1_S3"), (7, "R1_S0")]
                        if (f_id, f"R{r}_S{s}") not in valid_spots:
                            errors.append(f"Floor {f_id}: Unexpected Forensic call in {f'R{r}_S{s}'} (tag {tag})")

    # 3. Statistical Distribution
    print("--- TIER DISTRIBUTION SUMMARY ---")
    all_tiers = []
    for r in range(1, 5):
        for s in range(6):
            all_tiers.extend(df[f"R{r}_S{s}"].tolist())
    
    counts = pd.Series(all_tiers).value_counts()
    print(counts)

    if not errors:
        print("\n" + "="*40)
        print("[SUCCESS] 0 Logical Violations Found.")
        print("Dataset perfectly adheres to all project rules.")
        print("="*40)
    else:
        print("\n" + "!"*40)
        print(f"[FAILURE] Found {len(errors)} logical violations!")
        print("!"*40)
        for e in errors[:15]: print(f"  > {e}")

if __name__ == "__main__":
    run_integrity_audit()