# ==============================================================================
# Script: tools/parallel_worker.py
# Layer 4: Multiprocessing Utility
# Description: Houses the worker functions and the Grid Search algorithm. 
#              Abstracted so different optimizers (Max Floor, XP, Frags) can share it.
# ==============================================================================

import os
import sys
import random
import multiprocessing as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from engine.combat_loop import CombatSimulator
from tools.verify_player import load_state_from_json

JSON_PATH = os.path.join(BASE_DIR, "tools", "player_state.json")

def worker_simulate(payload):
    """Executes a single simulation instance. Payload contains stat distribution."""
    random.seed(os.urandom(4))
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
        
    for stat_name, val in payload['fixed_stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    
    sys.stdout = open(os.devnull, 'w')
    result = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    runtime_mins = result.total_time / 60.0
    
    return {
        "highest_floor": result.highest_floor,
        "total_xp": result.total_xp,
        "runtime_mins": runtime_mins,
        "xp_per_min": result.total_xp / runtime_mins if runtime_mins > 0 else 0,
        "frags": result.total_frags
    }

def generate_distributions(stats_list, total_budget, step, bounds=None):
    """Recursively generates valid stat combinations within boundaries."""
    distributions =[]
    def backtrack(idx, current_sum, current_dist):
        if idx == len(stats_list) - 1:
            remainder = total_budget - current_sum
            if bounds:
                min_v, max_v = bounds[stats_list[idx]]
                if not (min_v <= remainder <= max_v): return
            elif remainder < 0: return
                
            dist = current_dist.copy()
            dist[stats_list[idx]] = remainder
            distributions.append(dist)
            return
            
        stat_name = stats_list[idx]
        min_v = bounds[stat_name][0] if bounds else 0
        max_v = bounds[stat_name][1] if bounds else total_budget
        max_possible = min(max_v, total_budget - current_sum)
        
        for val in range(min_v, max_possible + 1, step):
            dist = current_dist.copy()
            dist[stat_name] = val
            backtrack(idx + 1, current_sum + val, dist)
            
    backtrack(0, 0, {})
    return distributions

def run_optimization_phase(phase_name, target_metric, stats_list, budget, step, iterations, pool, fixed_stats, bounds=None):
    """Runs a grid search phase and sorts by the requested target_metric (e.g., 'xp_per_min')."""
    dists = generate_distributions(stats_list, budget, step, bounds)
    if not dists: return None, None
        
    print(f"\n[{phase_name}] Step: {step} | Builds to test: {len(dists)} | Runs/Build: {iterations}")
    
    best_dist = None
    best_val = 0.0
    best_summary = None
    
    for i, dist in enumerate(dists):
        tasks =[{'stats': dist, 'fixed_stats': fixed_stats} for _ in range(iterations)]
        results = pool.map(worker_simulate, tasks)
        
        avg_xp_min = sum(r['xp_per_min'] for r in results) / iterations
        avg_floor = sum(r['highest_floor'] for r in results) / iterations
        
        # Determine the target we are optimizing for
        metric_val = avg_xp_min if target_metric == 'xp_per_min' else avg_floor
        
        if i % max(1, len(dists)//10) == 0 or metric_val > best_val:
            sys.stdout.write(f"\rProgress: {i+1}/{len(dists)} | Best {target_metric}: {best_val:,.0f}")
            sys.stdout.flush()
            
        if metric_val > best_val:
            best_val = metric_val
            best_dist = dist
            best_summary = {"avg_xp_min": avg_xp_min, "avg_floor": avg_floor}

    print(f"\n[{phase_name} Winner] {best_dist} -> {best_summary['avg_xp_min']:,.0f} XP/Min")
    return best_dist, best_summary