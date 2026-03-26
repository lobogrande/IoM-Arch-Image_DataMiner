import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import pandas as pd
import json

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

def audit_block_accuracy(mining_csv):
    df = pd.read_csv(mining_csv)
    boss_floors = cfg.BOSS_DATA.keys()
    
    results = []
    total_checks = 0
    correct_calls = 0

    print(f"--- BOSS TRUTH AUDIT: EVALUATING ACCURACY ---")

    for floor in boss_floors:
        # Get expected layout
        expected = {}
        if cfg.BOSS_DATA[floor].get('tier') == 'mixed':
            expected = cfg.BOSS_DATA[floor]['special']
        else:
            # Full floor of one tier
            expected = {i: cfg.BOSS_DATA[floor]['tier'] for i in range(24)}
        
        # Get Miner's calls for this floor (across all runs)
        for run_id in df['run'].unique():
            actual_calls = df[(df['run'] == run_id) & (df['floor'] == floor)]
            actual_map = {row['slot']: row['tier'] for _, row in actual_calls.iterrows()}
            
            for slot in range(24):
                total_checks += 1
                exp_tier = expected.get(slot) # May be None if slot is empty in mixed
                act_tier = actual_map.get(slot)
                
                if exp_tier == act_tier:
                    correct_calls += 1
                else:
                    results.append({
                        'run': run_id, 'floor': floor, 'slot': slot,
                        'expected': exp_tier if exp_tier else "empty",
                        'actual': act_tier if act_tier else "empty",
                        'error_type': "Mismatch" if (exp_tier and act_tier) else ("Miss" if exp_tier else "Ghost")
                    })

    accuracy = (correct_calls / total_checks) * 100
    print(f"\nFinal Accuracy Score: {accuracy:.2f}%")
    print(f"Total Errors Found: {len(results)}")
    
    error_df = pd.DataFrame(results)
    if not error_df.empty:
        print("\nTop Error Contributors (by Template):")
        print(error_df.groupby('actual').size().sort_values(ascending=False).head(5))
        error_df.to_csv("boss_truth_error_report.csv", index=False)
        print("\n[!] Detailed error report saved to: boss_truth_error_report.csv")

if __name__ == "__main__":
    audit_block_accuracy("archaeology_final_mining_data.csv")