# ==============================================================================
# Script: optimizers/opt_frag_farming.py
# Layer 4: Specific Optimizer (Fragment Farming)
# Description: Optimizes stat distribution to maximize fragments generated for a 
#              specific tier. Displays output in Banked Arch Seconds.
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

# ==========================================
# CONFIGURATION
# ==========================================
FRAG_NAMES = {
    0: "Dirt-Don't be silly, no one cares about dirt",
    1: "Common",
    2: "Rare",
    3: "Epic",
    4: "Legendary",
    5: "Mythic",
    6: "Divine"
}

# Change this integer (0-6) to target a different fragment type
TARGET_FRAG_TIER = 6

TARGET_METRIC = f"frag_{TARGET_FRAG_TIER}_per_min"
TARGET_NAME = FRAG_NAMES.get(TARGET_FRAG_TIER, "Unknown")
# ==========================================

if __name__ == "__main__":
    print(f"=== AI Arch Optimizer: Fragment Farming ({TARGET_NAME}) ===")
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
    if p.asc2_unlocked:
        STATS_TO_OPTIMIZE.append('Corr')
        
    DYNAMIC_BUDGET = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    cap_increase = int(p.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}

    print(f"Targeting Metric: {TARGET_METRIC} ({TARGET_NAME} Fragments)")
    print(f"Stats in Pool:    {STATS_TO_OPTIMIZE}")
    print(f"Total Budget:     {DYNAMIC_BUDGET} points")
    print(f"Effective Caps:   {EFFECTIVE_CAPS}\n")

    ITERATIONS_PER_DIST = 100
    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    start_time = time.time()
    
    with mp.Pool(CPU_CORES) as pool:
        step_1 = 10
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase(
            "Phase 1 (Coarse)", TARGET_METRIC, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_1, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p1
        )
        
        if not best_p1:
            print(f"\nOptimization Failed: Could not generate valid distributions or reach {TARGET_NAME} fragments.")
            sys.exit(1)
            
        bounds_p2 = {}
        for stat in STATS_TO_OPTIMIZE:
            val = best_p1[stat]
            bounds_p2[stat] = (max(0, val - step_1), min(EFFECTIVE_CAPS[stat], val + step_1))
            
        step_2 = 3
        best_p2, _ = run_optimization_phase(
            "Phase 2 (Fine)", TARGET_METRIC, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_2, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p2
        )
        
        bounds_p3 = {}
        if best_p2:
            for stat in STATS_TO_OPTIMIZE:
                val = best_p2[stat]
                bounds_p3[stat] = (max(0, val - step_2), min(EFFECTIVE_CAPS[stat], val + step_2))
        else:
            bounds_p3 = bounds_p2

        best_p3, final_summary = run_optimization_phase(
            "Phase 3 (Exact)", TARGET_METRIC, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, 1, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p3
        )
        
    elapsed = time.time() - start_time
    
    # --- UX: Banked Time Conversions ---
    rate_per_min = final_summary[TARGET_METRIC]
    rate_per_sec = rate_per_min / 60.0
    rate_per_1k_secs = rate_per_sec * 1000.0
    
    print("\n" + "="*50)
    print(f"Optimization Complete in {elapsed:.2f} seconds.")
    print(f"Best Stat Build to Farm {TARGET_NAME} Fragments:")
    for stat in STATS_TO_OPTIMIZE:
        print(f"  {stat}: {best_p3[stat]}")
    for k, v in FIXED_STATS.items():
        print(f"  {k}: {v} (Fixed)")
        
    print(f"\n[ RATE PROJECTIONS ]")
    print(f" - Real-Time:   {rate_per_min:,.2f} {TARGET_NAME} Frags / Minute")
    print(f" - Banked Time: {rate_per_sec:,.2f} {TARGET_NAME} Frags / Arch Second")
    print(f" - Banked Time: {rate_per_1k_secs:,.1f} {TARGET_NAME} Frags / 1k Arch Seconds")
    print("="*50)