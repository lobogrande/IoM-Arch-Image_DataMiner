# step7_integrity_audit.py
# Purpose: Verify the logical and mathematical integrity of the 110-floor dataset.
# Version: 2.0 (Generalized for Multi-Dataset Runs)

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_integrity_audit():
    # Resolve dynamic path
    source_dir = cfg.get_buffer_path()
    run_id = os.path.basename(source_dir).split('_')[-1]
    csv_path = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_block_inventory_run_{run_id}.csv")
    
    if not os.path.exists(csv_path):
        print(f"!!! ERROR: Could not find inventory at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    errors =[]
    
    print(f"--- STEP 7: FINAL DATASET AUDIT (Run {run_id}) ---")
    
    forensic_count = 0
    total_occupied_slots = 0
    
    for _, row in df.iterrows():
        f_id = int(row['floor_id'])
        
        # 1. Violation Check: ORE_RESTRICTIONS
        for r in range(1, 5):
            for s in range(6):
                slot_key = f"R{r}_S{s}"
                tier = str(row[slot_key])
                
                if tier != "empty":
                    total_occupied_slots += 1
                    
                if tier in["empty", "low_conf", "likely_empty", "obstructed", "unknown_obstructed"]: 
                    continue
                
                limit = cfg.ORE_RESTRICTIONS.get(tier)
                if limit and not (limit[0] <= f_id <= limit[1]):
                    errors.append(f"Floor {f_id}: {tier} found in {slot_key} (FORBIDDEN by biome restrictions)")

        # 2. Forensic Check (Generalized for Runs 1-N)
        for r in range(1, 5):
            for s in range(6):
                tag_col = f"R{r}_S{s}_tag"
                if tag_col in row:
                    tag = str(row[tag_col])
                    if tag in ["[L]", "[U]", "[O]"]:
                        forensic_count += 1

    # 3. Statistical Distribution
    print("--- TIER DISTRIBUTION SUMMARY ---")
    all_tiers =[]
    for r in range(1, 5):
        for s in range(6):
            all_tiers.extend(df[f"R{r}_S{s}"].tolist())
    
    counts = pd.Series(all_tiers).value_counts()
    print(counts)

    # 4. Forensic Over-Saturation Guardrail
    # If more than 2% of all occupied slots are flagged as totally obstructed, 
    # the homing script (Step 1) is likely mis-calibrated for this dataset.
    max_allowed_forensics = int(total_occupied_slots * 0.02)
    if forensic_count > max_allowed_forensics:
        errors.append(f"Too many obstructed slots! Found {forensic_count} (Max allowed: {max_allowed_forensics}). Check Step 1 calibration.")

    if not errors:
        print("\n" + "="*45)
        print(f"[SUCCESS] 0 Logical Violations Found.")
        print(f"Forensic Obstructions: {forensic_count} (Well within {max_allowed_forensics} limit).")
        print("Dataset perfectly adheres to all project rules.")
        print("="*45)
    else:
        print("\n" + "!"*45)
        print(f"[FAILURE] Found {len(errors)} logical violations!")
        print("!"*45)
        for e in errors[:15]: print(f"  > {e}")

if __name__ == "__main__":
    run_integrity_audit()