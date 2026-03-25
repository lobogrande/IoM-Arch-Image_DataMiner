# ==============================================================================
# Script: tools/opt_frag_farming.py
# Layer 4: Specific Optimizer (Fragment Farming)
# Description: Optimizes stat distribution to maximize Divinity (or Mythic) 
#              fragments generated per minute of combat.
# ==============================================================================

import os, sys, time
import multiprocessing as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
from tools.parallel_worker import run_optimization_phase, JSON_PATH
from project_config import BASE_STAT_CAPS

if __name__ == "__main__":
    print("=== AI Arch Optimizer: Divinity Fragment Farming ===")
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
    if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
        
    DYNAMIC_BUDGET = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    cap_increase = int(p.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}

    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(CPU_CORES) as pool:
        # Target metric: frag_6_per_min (Divinity). 
        # For Mythic, change this to 'frag_5_per_min'
        target_metric = 'frag_6_per_min'
        
        step_1 = 10
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase("Phase 1", target_metric, STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, step_1, 100, pool, FIXED_STATS, bounds_p1)
        
        bounds_p2 = {s: (max(0, best_p1[s] - step_1), min(EFFECTIVE_CAPS[s], best_p1[s] + step_1)) for s in STATS_TO_OPTIMIZE} if best_p1 else {}
        best_p2, _ = run_optimization_phase("Phase 2", target_metric, STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, 3, 100, pool, FIXED_STATS, bounds_p2)
        
        bounds_p3 = {s: (max(0, best_p2[s] - 3), min(EFFECTIVE_CAPS[s], best_p2[s] + 3)) for s in STATS_TO_OPTIMIZE} if best_p2 else bounds_p2
        best_p3, summary = run_optimization_phase("Phase 3", target_metric, STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, 1, 100, pool, FIXED_STATS, bounds_p3)
        
    print("\nOptimal Frag Farming Build:", best_p3)