# audit_final_results.py
# Purpose: Verify the logical and mathematical integrity of the 110-floor dataset.

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_integrity_audit(csv_path):
    df = pd.read_csv(csv_path)
    errors = []
    
    print(f"--- STARTING FINAL DATASET AUDIT ---")
    
    for _, row in df.iterrows():
        f_id = int(row['floor_id'])
        
        # 1. Violation Check: ORE_RESTRICTIONS
        # Ensure no tier appears on a floor where it is forbidden
        for r in range(1, 5):
            for s in range(6):
                slot_key = f"R{r}_S{s}"
                tier = str(row[slot_key])
                if tier in ["empty", "low_conf", "likely_empty", "obstructed"]: continue
                
                limit = cfg.ORE_RESTRICTIONS.get(tier)
                if limit and not (limit[0] <= f_id <= limit[1]):
                    errors.append(f"Floor {f_id}: {tier} found in {slot_key} (FORBIDDEN by restrictions)")

        # 2. Forensic Check: The "Cyan/Yellow" Rule
        # Per your requirement, these should ONLY exist in specific spots
        for r in range(1, 5):
            for s in range(6):
                tag = str(row.get(f"R{r}_S{s}_tag", ""))
                if tag in ["[L]", "[U]"]:
                    # Rule: Only F2 (S1), F3 (S1), F7 (S0)
                    valid_spots = [(2, "R1_S1"), (3, "R1_S1"), (7, "R1_S0")]
                    if (f_id, f"R{r}_S{s}") not in valid_spots:
                        errors.append(f"Floor {f_id}: Unexpected Forensic call in {f'R{r}_S{s}'} (tag {tag})")

    # 3. Statistical Distribution
    print("\n--- TIER DISTRIBUTION SUMMARY ---")
    all_tiers = []
    for r in range(1, 5):
        for s in range(6):
            all_tiers.extend(df[f"R{r}_S{s}"].tolist())
    
    counts = pd.Series(all_tiers).value_counts()
    print(counts)

    if not errors:
        print("\n[SUCCESS] 0 Logical Violations Found. Dataset adheres to all project rules.")
    else:
        print(f"\n[FAILURE] Found {len(errors)} logical violations!")
        for e in errors[:10]: print(f"  > {e}")

if __name__ == "__main__":
    run_integrity_audit("floor_ore_inventory.csv")