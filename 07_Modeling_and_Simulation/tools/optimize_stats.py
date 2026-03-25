# ==============================================================================
# Script: tools/optimize_stats.py
# Layer 4: Monte Carlo Optimizer (Zoom-In Multi-Phase Search w/ Caps)
# Description: Dynamically reads total stat budget and upgrade levels from JSON. 
#              Applies rigorous stat caps (Base + Upgrade 45). Uses a 3-phase 
#              zooming grid search to find the mathematically perfect stat distribution.
# ==============================================================================

import os
import sys
import random
import multiprocessing as mp
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from engine.combat_loop import CombatSimulator
from tools.verify_player import load_state_from_json

JSON_PATH = os.path.join(BASE_DIR, "tools", "player_state.json")

# Which stats do we want the optimizer to play with? 
STATS_TO_OPTIMIZE = ['Str', 'Agi', 'Luck', 'Div'] 

# Base caps before any upgrades
BASE_STAT_CAPS = {
    'Str': 50,
    'Agi': 50,
    'Per': 25,
    'Int': 25,
    'Luck': 25,
    'Div': 10,
    'Corr': 10
}

def worker_simulate(payload):
    """Top-level worker function to execute CombatSimulator safely in parallel."""
    random.seed(os.urandom(4)) # Prevent RNG lock
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    # Apply dynamic optimized stats
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
        
    # Ensure non-optimized stats remain untouched
    for stat_name, val in payload['fixed_stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    
    # Suppress console spam from the engine
    sys.stdout = open(os.devnull, 'w')
    result_state = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    runtime_mins = result_state.total_time / 60.0
    xp_per_min = result_state.total_xp / runtime_mins if runtime_mins > 0 else 0
    
    return {
        "highest_floor": result_state.highest_floor,
        "total_xp": result_state.total_xp,
        "runtime_mins": runtime_mins,
        "xp_per_min": xp_per_min
    }

def generate_distributions(stats_list, total_budget, step, bounds=None):
    """
    Recursively generates all valid stat combinations summing to total_budget.
    Respects strict min/max caps dictated by the bounds argument.
    """
    distributions =[]
    
    def backtrack(idx, current_sum, current_dist):
        if idx == len(stats_list) - 1:
            remainder = total_budget - current_sum
            # Check if remainder fits within the bounds for the final stat
            if bounds:
                min_v, max_v = bounds[stats_list[idx]]
                if not (min_v <= remainder <= max_v):
                    return
            elif remainder < 0:
                return
                
            dist = current_dist.copy()
            dist[stats_list[idx]] = remainder
            distributions.append(dist)
            return
            
        stat_name = stats_list[idx]
        min_v = bounds[stat_name][0] if bounds else 0
        max_v = bounds[stat_name][1] if bounds else total_budget
        
        max_possible = min(max_v, total_budget - current_sum)
        
        # Start at min_v, step forward
        for val in range(min_v, max_possible + 1, step):
            dist = current_dist.copy()
            dist[stat_name] = val
            backtrack(idx + 1, current_sum + val, dist)
            
    backtrack(0, 0, {})
    return distributions

def run_optimization_phase(phase_name, stats_list, budget, step, iterations, pool, fixed_stats, bounds=None):
    """Runs a single phase of the grid search and returns the best distribution."""
    dists = generate_distributions(stats_list, budget, step, bounds)
    
    if not dists:
        print(f"[{phase_name}] No valid combinations found with current bounds. Falling back...")
        return None, None
        
    print(f"\n--- {phase_name} ---")
    print(f"Step Size: {step} | Testing {len(dists)} unique builds | Runs per build: {iterations}")
    
    best_dist = None
    best_xp_val = 0.0
    best_summary = None
    
    for i, dist in enumerate(dists):
        tasks =[{'stats': dist, 'fixed_stats': fixed_stats} for _ in range(iterations)]
        results = pool.map(worker_simulate, tasks)
        
        avg_xp_min = sum(r['xp_per_min'] for r in results) / iterations
        avg_floor = sum(r['highest_floor'] for r in results) / iterations
        
        # Log progress for top 10% of runs to keep terminal clean
        if i % max(1, len(dists)//10) == 0 or avg_xp_min > best_xp_val:
            sys.stdout.write(f"\rProgress: {i+1}/{len(dists)} | Current Best XP/Min: {best_xp_val:,.0f}")
            sys.stdout.flush()
            
        if avg_xp_min > best_xp_val:
            best_xp_val = avg_xp_min
            best_dist = dist
            best_summary = {"avg_xp_min": avg_xp_min, "avg_floor": avg_floor}

    print(f"\n[{phase_name} Winner] {best_dist} -> {best_summary['avg_xp_min']:,.0f} XP/Min")
    return best_dist, best_summary

if __name__ == "__main__":
    print("=== AI Arch Monte Carlo Optimizer (Zoom-In Search w/ Caps) ===")
    
    # 1. Dynamically read budget and state from JSON
    temp_player = Player()
    load_state_from_json(temp_player, JSON_PATH)
    
    DYNAMIC_BUDGET = int(sum(temp_player.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in temp_player.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    # 2. Calculate Strict Caps based on Base Caps + Upgrade #45
    cap_increase = int(temp_player.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}
    
    max_possible_points = sum(EFFECTIVE_CAPS[s] for s in STATS_TO_OPTIMIZE)
    
    print(f"Target Stats to Optimize: {STATS_TO_OPTIMIZE}")
    print(f"Dynamic Budget Available: {DYNAMIC_BUDGET} points")
    print(f"Cap Increases Unlocked:   +{cap_increase}")
    print(f"Strict Stat Caps:         {EFFECTIVE_CAPS}")
    print(f"Fixed Stats (Ignored):    {FIXED_STATS}\n")
    
    if DYNAMIC_BUDGET > max_possible_points:
        print(f"ERROR: Budget ({DYNAMIC_BUDGET}) exceeds the sum of stat caps ({max_possible_points}).")
        sys.exit(1)
    
    ITERATIONS_PER_DIST = 100
    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    start_time = time.time()
    
    with mp.Pool(CPU_CORES) as pool:
        # PHASE 1: Coarse Search (Jump by 15)
        step_1 = 15
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase(
            "Phase 1: Coarse Search", STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, 
            step_1, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p1
        )
        
        # Setup bounds for Phase 2 based on Phase 1 winner (Clamp to Effective Caps)
        bounds_p2 = {}
        if best_p1:
            for stat in STATS_TO_OPTIMIZE:
                val = best_p1[stat]
                bounds_p2[stat] = (max(0, val - step_1), min(EFFECTIVE_CAPS[stat], val + step_1))
            
        # PHASE 2: Fine Search (Jump by 3)
        step_2 = 3
        best_p2, _ = run_optimization_phase(
            "Phase 2: Fine Search", STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, 
            step_2, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p2
        )
        
        # Setup bounds for Phase 3 based on Phase 2 winner (Clamp to Effective Caps)
        bounds_p3 = {}
        if best_p2:
            for stat in STATS_TO_OPTIMIZE:
                val = best_p2[stat]
                bounds_p3[stat] = (max(0, val - step_2), min(EFFECTIVE_CAPS[stat], val + step_2))
        else:
            bounds_p3 = bounds_p2 # Fallback
            best_p2 = best_p1

        # PHASE 3: Exact Search (Jump by 1)
        step_3 = 1
        best_p3, final_summary = run_optimization_phase(
            "Phase 3: Exact 1-Point Search", STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, 
            step_3, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p3
        )
        
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("=== FINAL OPTIMIZED BUILD FOUND ===")
    print("="*50)
    print(f"Total Computation Time: {elapsed:.2f} seconds")
    print("\nOPTIMAL STAT ALLOCATION:")
    
    final_best = best_p3 if best_p3 else best_p2
    for stat in STATS_TO_OPTIMIZE:
        print(f" - {stat}: {final_best[stat]} (Cap: {EFFECTIVE_CAPS[stat]})")
    for k, v in FIXED_STATS.items():
        print(f" - {k}: {v} (Fixed)")
        
    if final_summary:
        print(f"\nPROJECTED METRICS:")
        print(f" - Average XP/Min:  {final_summary['avg_xp_min']:,.0f}")
        print(f" - Average Floor:   {final_summary['avg_floor']:.1f}")