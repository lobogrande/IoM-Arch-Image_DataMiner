# ==============================================================================
# Script: tools/parallel_worker.py
# Layer 4: Multiprocessing Utility & Engine (Successive Halving + Live Progress)
# Description: Houses the worker functions, grid search algorithm, Successive 
#              Halving (Early Stopping) logic, and ETA hardware benchmarking.
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
    
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
        
    for stat_name, val in payload['fixed_stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    
    sys.stdout = open(os.devnull, 'w')
    result = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    runtime_mins = result.total_time / 60.0 if result.total_time > 0 else 1.0
    
    metrics = {
        "highest_floor": result.highest_floor,
        "xp_per_min": result.total_xp / runtime_mins,
        "ores_per_min": result.ores_mined / runtime_mins
    }
    
    for frag_tier, amt in result.total_frags.items():
        metrics[f"frag_{frag_tier}_per_min"] = amt / runtime_mins
        
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
    """
    Runs a grid search phase using Successive Halving with live progress tracking.
    """
    dists = generate_distributions(stats_list, budget, step, bounds)
    if not dists: return None, None
        
    print(f"\n[{phase_name}] Step: {step} | Total Initial Builds: {len(dists)}")
    
    tracker = {}
    for d in dists:
        key = tuple(sorted(d.items()))
        tracker[key] = {'dist': d, 'sum_target': 0.0, 'sum_floor': 0.0, 'runs': 0}
        
    active_keys = list(tracker.keys())
    
    if len(dists) <= 20 or iterations <= 10:
        rounds = [(iterations, 1.0)] 
    else:
        r1 = max(1, int(iterations * 0.15))
        r2 = max(1, int(iterations * 0.35))
        r3 = iterations - r1 - r2
        rounds =[
            (r1, 0.20),
            (r2, 0.10),
            (r3, 1.0)
        ]

    for round_idx, (run_count, keep_ratio) in enumerate(rounds):
        if len(active_keys) == 0: break
        
        if len(rounds) > 1:
            print(f"  -> Round {round_idx+1}: Testing {len(active_keys)} builds ({run_count} runs each)...")
            
        # Build payload tasks
        tasks = [{'stats': tracker[k]['dist'], 'fixed_stats': fixed_stats} for k in active_keys for _ in range(run_count)]
        total_tasks = len(tasks)
        
        # Determine efficient chunksize for multiprocessing IPC
        chunk_size = max(1, total_tasks // 100)
        
        results =[]
        # Use imap to yield results continuously, allowing us to update the terminal
        for i, r in enumerate(pool.imap(worker_simulate, tasks, chunksize=chunk_size)):
            results.append(r)
            
            # Print progress roughly 20 times (every 5%) to avoid terminal flicker/lag
            if i % max(1, total_tasks // 20) == 0 or i == total_tasks - 1:
                sys.stdout.write(f"\r      Progress: {i+1}/{total_tasks} simulations completed")
                sys.stdout.flush()
                
        sys.stdout.write("\n") # Drop to a new line after the progress bar completes
        
        # Aggregate results
        for i, k in enumerate(active_keys):
            chunk = results[i*run_count : (i+1)*run_count]
            tracker[k]['sum_target'] += sum(r.get(target_metric, 0.0) for r in chunk)
            tracker[k]['sum_floor'] += sum(r.get('highest_floor', 0.0) for r in chunk)
            tracker[k]['runs'] += run_count
            
        # Sort and cull
        active_keys.sort(key=lambda k: tracker[k]['sum_target'] / tracker[k]['runs'], reverse=True)
        
        if round_idx < len(rounds) - 1:
            keep_count = max(3, int(len(active_keys) * keep_ratio))
            active_keys = active_keys[:keep_count]

    best_key = active_keys[0]
    best_data = tracker[best_key]
    
    best_dist = best_data['dist']
    best_summary = {
        target_metric: best_data['sum_target'] / best_data['runs'],
        "avg_floor": best_data['sum_floor'] / best_data['runs']
    }

    if best_summary[target_metric] > 0:
        print(f"[{phase_name} Winner] {best_dist} -> {best_summary[target_metric]:,.2f} {target_metric}")
    else:
        print(f"[{phase_name}] Target metric '{target_metric}' not found.")
    
    return best_dist, best_summary

# --- HARDWARE BENCHMARKING & ETA PREDICTION ---

def benchmark_hardware(baseline_payload, pool, test_iterations=200):
    """Runs a fast micro-benchmark to determine CPU processing speed."""
    tasks =[baseline_payload for _ in range(test_iterations)]
    start_time = time.time()
    pool.map(worker_simulate, tasks)
    elapsed = time.time() - start_time
    return test_iterations / elapsed if elapsed > 0 else 1

def get_eta_profiles(stats_list, budget, caps, sims_per_second, iterations_per_build=100):
    """Calculates completion ETAs based on the hardware benchmark and halving logic."""
    profiles = {
        "Fast (Step: 15)": {"step": 15},
        "Standard (Step: 10)": {"step": 10},
        "Deep (Step: 5)": {"step": 5}
    }
    bounds = {s: (0, caps[s]) for s in stats_list}
    
    for name, data in profiles.items():
        p1_builds = len(generate_distributions(stats_list, budget, data["step"], bounds))
        total_estimated_builds = p1_builds + 300 
        
        total_simulations = (total_estimated_builds * iterations_per_build) * 0.25
        estimated_seconds = total_simulations / sims_per_second
        
        if estimated_seconds < 60:
            time_str = f"~{int(estimated_seconds)} seconds"
        else:
            time_str = f"~{estimated_seconds/60:.1f} minutes"
            
        data["builds"] = total_estimated_builds
        data["eta_seconds"] = estimated_seconds
        data["time_label"] = time_str
        
    return profiles