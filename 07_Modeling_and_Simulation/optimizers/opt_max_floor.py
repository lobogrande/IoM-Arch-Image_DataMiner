# ==============================================================================
# Script: optimizers/opt_max_floor.py
# Layer 4: Specific Optimizer (Max Floor Push)
# Description: Optimizes stat distribution to reach the highest possible floor
#              before stamina depletion. Focuses on the Absolute Max "God Run".
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
from optimizers.parallel_worker import run_optimization_phase, JSON_PATH
from project_config import BASE_STAT_CAPS

if __name__ == "__main__":
    print("=== AI Arch Optimizer: Maximum Floor Push ===")
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    STATS_TO_OPTIMIZE = ['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
    if p.asc2_unlocked:
        STATS_TO_OPTIMIZE.append('Corr')
        
    DYNAMIC_BUDGET = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    cap_increase = int(p.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}

    print(f"Targeting Metric: highest_floor (Absolute Maximum 'God Run')")
    print(f"Stats in Pool:    {STATS_TO_OPTIMIZE}")
    print(f"Total Budget:     {DYNAMIC_BUDGET} points")
    print(f"Effective Caps:   {EFFECTIVE_CAPS}\n")

    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    # We use progressive scaling, but push Phase 3 heavily to ensure we catch the RNG outliers
    ITER_P1, ITER_P2, ITER_P3 = 25, 50, 200 
    
    start_time = time.time()
    
    with mp.Pool(CPU_CORES) as pool:
        target_metric = 'highest_floor'
        
        step_1 = 10
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase(
            "Phase 1 (Coarse)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_1, ITER_P1, pool, FIXED_STATS, bounds_p1
        )
        
        if not best_p1:
            print(f"\nOptimization Failed: Could not generate valid distributions.")
            sys.exit(1)
            
        bounds_p2 = {}
        for stat in STATS_TO_OPTIMIZE:
            val = best_p1[stat]
            bounds_p2[stat] = (max(0, val - step_1), min(EFFECTIVE_CAPS[stat], val + step_1))
            
        step_2 = 3
        best_p2, _ = run_optimization_phase(
            "Phase 2 (Fine)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_2, ITER_P2, pool, FIXED_STATS, bounds_p2
        )
        
        bounds_p3 = {}
        if best_p2:
            p3_radius = min(2, step_2)
            for stat in STATS_TO_OPTIMIZE:
                val = best_p2[stat]
                bounds_p3[stat] = (max(0, val - p3_radius), min(EFFECTIVE_CAPS[stat], val + p3_radius))
        else:
            bounds_p3 = bounds_p2

        best_p3, final_summary = run_optimization_phase(
            "Phase 3 (Exact)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, 1, ITER_P3, pool, FIXED_STATS, bounds_p3
        )
        
    elapsed = time.time() - start_time
    
    # --- UX: Final Output Readout ---
    abs_max = final_summary['abs_max_floor']
    abs_chance = final_summary['abs_max_chance'] * 100
    avg_floor = final_summary['avg_floor']
    
    print("\n" + "="*50)
    print(f"Optimization Complete in {elapsed:.2f} seconds.")
    print("Best Stat Build to Push Max Floor:")
    for stat in STATS_TO_OPTIMIZE:
        print(f"  {stat}: {best_p3[stat]}")
    for k, v in FIXED_STATS.items():
        print(f"  {k}: {v} (Fixed)")
        
    print(f"\n[ PERFORMANCE PROJECTION ]")
    print(f" - Absolute Maximum Floor Reached: Floor {abs_max}!!!")
    print(f" - Likelihood of God-Run:          {abs_chance:.1f}% ({int(final_summary['abs_max_chance'] * ITER_P3)} in {ITER_P3} runs)")
    print(f" - Average Consistency Floor:      Floor {avg_floor:,.1f}")
    print("="*50)