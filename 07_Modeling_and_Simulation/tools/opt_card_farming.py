# ==============================================================================
# Script: tools/opt_card_farming.py
# Layer 4: Specific Optimizer (Card/Ore Farming)
# Description: Optimizes stat distribution to maximize kills per minute of a 
#              SPECIFIC ore type/tier (e.g., maximizing 'myth3' card drops).
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
from tools.parallel_worker import run_optimization_phase, JSON_PATH
from project_config import BASE_STAT_CAPS

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to the exact ore you want to farm.
# Format: {type}{tier}. Examples: 'dirt1', 'com2', 'myth3', 'div4'
TARGET_ORE = "myth3"

# Automatically translates into the dictionary key returned by the parallel worker
TARGET_METRIC = f"ore_{TARGET_ORE}_per_min"
# ==========================================

if __name__ == "__main__":
    print(f"=== AI Arch Optimizer: Card Farming ({TARGET_ORE}) ===")
    
    # 1. Load Player State
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    # 2. Determine Available Stats
    STATS_TO_OPTIMIZE = ['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
    if p.asc2_unlocked:
        STATS_TO_OPTIMIZE.append('Corr')
        
    DYNAMIC_BUDGET = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    # 3. Calculate Strict Caps
    cap_increase = int(p.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}
    
    print(f"Targeting Metric: {TARGET_METRIC}")
    print(f"Stats in Pool:    {STATS_TO_OPTIMIZE}")
    print(f"Total Budget:     {DYNAMIC_BUDGET} points")
    print(f"Effective Caps:   {EFFECTIVE_CAPS}\n")

    ITERATIONS_PER_DIST = 100
    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    start_time = time.time()
    
    with mp.Pool(CPU_CORES) as pool:
        # Phase 1: Coarse Search (Step 10)
        step_1 = 10
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase(
            "Phase 1 (Coarse)", TARGET_METRIC, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_1, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p1
        )
        
        if not best_p1:
            print("\nOptimization Failed: Could not generate valid distributions or reach target ore.")
            sys.exit(1)
            
        # Phase 2: Fine Search (Step 3) - Constrain around Phase 1 winner
        bounds_p2 = {}
        for stat in STATS_TO_OPTIMIZE:
            val = best_p1[stat]
            bounds_p2[stat] = (max(0, val - step_1), min(EFFECTIVE_CAPS[stat], val + step_1))
            
        step_2 = 3
        best_p2, _ = run_optimization_phase(
            "Phase 2 (Fine)", TARGET_METRIC, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_2, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p2
        )
        
        # Phase 3: Exact Search (Step 1) - Constrain around Phase 2 winner
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
    print("\n" + "="*50)
    print(f"Optimization Complete in {elapsed:.2f} seconds.")
    print(f"Best Stat Build to Farm {TARGET_ORE}:")
    for stat in STATS_TO_OPTIMIZE:
        print(f"  {stat}: {best_p3[stat]}")
    for k, v in FIXED_STATS.items():
        print(f"  {k}: {v} (Fixed)")
    print(f"\nProjected Rate: {final_summary[TARGET_METRIC]:,.2f} {TARGET_ORE} kills per minute")
    print("="*50)