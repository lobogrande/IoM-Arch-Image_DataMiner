# ==============================================================================
# Script: tools/parallel_worker.py
# Layer 4: Multiprocessing Utility & Engine
# Description: Houses the worker functions, grid search algorithm, and ETA 
#              hardware benchmarking. Abstracted so different optimizers can share it.
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

def worker_simulate(payload):
    """Executes a single simulation instance. Payload contains stat distribution."""
    random.seed(os.urandom(4))
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    # Inject optimized stats
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
        
    # Inject locked/fixed stats
    for stat_name, val in payload['fixed_stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    
    # Suppress output to keep terminal clean during multiprocessing
    sys.stdout = open(os.devnull, 'w')
    result = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    runtime_mins = result.total_time / 60.0 if result.total_time > 0 else 1.0
    
    # 1. Base Metrics
    metrics = {
        "highest_floor": result.highest_floor,
        "xp_per_min": result.total_xp / runtime_mins,
        "ores_per_min": result.ores_mined / runtime_mins
    }
    
    # 2. Fragment Tier Metrics
    for frag_tier, amt in result.total_frags.items():
        metrics[f"frag_{frag_tier}_per_min"] = amt / runtime_mins
        
    # 3. Specific Ore Farming Metrics (Requires the telemetry update in combat_loop.py)
    if hasattr(result, 'specific_ores_mined'):
        for ore_id, count in result.specific_ores_mined.items():
            metrics[f"ore_{ore_id}_per_min"] = count / runtime_mins
            
    return metrics

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
    """Runs a grid search phase and sorts dynamically by the requested target_metric."""
    dists = generate_distributions(stats_list, budget, step, bounds)
    if not dists: return None, None
        
    print(f"\n[{phase_name}] Step: {step} | Builds to test: {len(dists)} | Runs/Build: {iterations}")
    
    best_dist = None
    best_val = 0.0
    best_summary = None
    
    for i, dist in enumerate(dists):
        tasks =[{'stats': dist, 'fixed_stats': fixed_stats} for _ in range(iterations)]
        results = pool.map(worker_simulate, tasks)
        
        # Dynamically average whatever metric the optimizer script asked for
        # Uses .get() with a fallback of 0.0 so the script doesn't crash if an ore wasn't encountered
        avg_target = sum(r.get(target_metric, 0.0) for r in results) / iterations
        avg_floor = sum(r['highest_floor'] for r in results) / iterations
        
        if i % max(1, len(dists)//10) == 0 or avg_target > best_val:
            sys.stdout.write(f"\rProgress: {i+1}/{len(dists)} | Best {target_metric}: {best_val:,.2f}")
            sys.stdout.flush()
            
        if avg_target > best_val:
            best_val = avg_target
            best_dist = dist
            best_summary = {target_metric: avg_target, "avg_floor": avg_floor}

    if best_dist:
        print(f"\n[{phase_name} Winner] {best_dist} -> {best_summary[target_metric]:,.2f} {target_metric}")
    else:
        print(f"\n[{phase_name}] Target metric '{target_metric}' not found during runs. Ensure your player can reach the target.")
    
    return best_dist, best_summary

# --- HARDWARE BENCHMARKING & ETA PREDICTION ---

def benchmark_hardware(baseline_payload, pool, test_iterations=200):
    """Runs a fast micro-benchmark to determine CPU processing speed."""
    tasks = [baseline_payload for _ in range(test_iterations)]
    
    start_time = time.time()
    pool.map(worker_simulate, tasks)
    elapsed = time.time() - start_time
    
    sims_per_second = test_iterations / elapsed if elapsed > 0 else 1
    return sims_per_second

def get_eta_profiles(stats_list, budget, caps, sims_per_second, iterations_per_build=100):
    """Calculates completion ETAs based on the hardware benchmark."""
    profiles = {
        "Fast (Step: 15)": {"step": 15},
        "Standard (Step: 10)": {"step": 10},
        "Deep (Step: 5)": {"step": 5}
    }
    
    bounds = {s: (0, caps[s]) for s in stats_list}
    
    for name, data in profiles.items():
        p1_builds = len(generate_distributions(stats_list, budget, data["step"], bounds))
        total_estimated_builds = p1_builds + 300 # Buffer for Phase 2 and 3
        
        total_simulations = total_estimated_builds * iterations_per_build
        estimated_seconds = total_simulations / sims_per_second
        
        if estimated_seconds < 60:
            time_str = f"~{int(estimated_seconds)} seconds"
        else:
            time_str = f"~{estimated_seconds/60:.1f} minutes"
            
        data["builds"] = total_estimated_builds
        data["eta_seconds"] = estimated_seconds
        data["time_label"] = time_str
        
    return profiles