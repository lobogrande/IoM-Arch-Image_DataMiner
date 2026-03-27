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
    
    # --- IN-MEMORY STATE RECONSTRUCTION (ZERO DISK I/O) ---
    state = payload.get('state_dict', {})
    if state:
        p.base_stats = state.get('base_stats', {}).copy()
        p.asc2_unlocked = state.get('asc2_unlocked', False)
        p.arch_level = state.get('arch_level', 1)
        p.current_max_floor = state.get('current_max_floor', 1)
        p.hades_idol_level = state.get('hades_idol_level', 0)
        p.arch_ability_infernal_bonus = state.get('arch_ability_infernal_bonus', 0.0)
        p.total_infernal_cards = state.get('total_infernal_cards', 0)
        
        # PROPERLY RE-APPLY UPGRADES USING SETTERS SO MATH TRIGGERS!
        for upg_id, lvl in state.get('upgrade_levels', {}).items():
            p.set_upgrade_level(upg_id, lvl)
        for ext_id, lvl in state.get('external_levels', {}).items():
            p.set_external_level(ext_id, lvl)
        for card_id, lvl in state.get('cards', {}).items():
            p.set_card_level(card_id, lvl)
    else:
        # Fallback for local CLI script testing
        load_state_from_json(p, JSON_PATH)
    
    # Inject the specific stat distribution being tested by the grid search
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
        "blocks_per_min": result.blocks_mined / runtime_mins
    }
    
    for frag_tier, amt in result.total_frags.items():
        metrics[f"frag_{frag_tier}_per_min"] = amt / runtime_mins
        
    if hasattr(result, 'specific_blocks_mined'):
        for block_id, count in result.specific_blocks_mined.items():
            metrics[f"block_{block_id}_per_min"] = count / runtime_mins
            
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

def run_optimization_phase(phase_name, target_metric, stats_list, budget, step, iterations, pool, fixed_stats, bounds=None, progress_callback=None, global_start_time=None, time_limit_seconds=None, base_state_dict=None):
    """
    Runs a grid search phase using Successive Halving with live progress tracking.
    """
    dists = generate_distributions(stats_list, budget, step, bounds)
    if not dists: return None, None
        
    print(f"\n[{phase_name}] Step: {step} | Total Initial Builds: {len(dists)}")
    
    tracker = {}
    for d in dists:
        key = tuple(sorted(d.items()))
        # Added 'metrics_sum' and 'floors' to aggregate deep telemetry for the UI
        tracker[key] = {'dist': d, 'sum_target': 0.0, 'sum_floor': 0.0, 'runs': 0, 'metrics_sum': {}, 'floors':[]}
        
    active_keys = list(tracker.keys())
    
    if len(dists) <= 20 or iterations <= 10:
        rounds = [(iterations, 1.0)] 
    else:
        r1 = max(1, int(iterations * 0.15))
        r2 = max(1, int(iterations * 0.35))
        r3 = iterations - r1 - r2
        rounds =[(r1, 0.20), (r2, 0.10), (r3, 1.0)]

    for round_idx, (run_count, keep_ratio) in enumerate(rounds):
        if len(active_keys) == 0: break
        
        if global_start_time and time_limit_seconds:
            if time.time() - global_start_time >= time_limit_seconds:
                print(f"\n[TIMEOUT] Max time limit reached. Halting {phase_name} early!")
                break
        
        if len(rounds) > 1:
            print(f"  -> Round {round_idx+1}: Testing {len(active_keys)} builds ({run_count} runs each)...")
            
        tasks = [{'stats': tracker[k]['dist'], 'fixed_stats': fixed_stats, 'state_dict': base_state_dict} for k in active_keys for _ in range(run_count)]
        total_tasks = len(tasks)
        chunk_size = max(1, total_tasks // 100)
        
        results =[]
        for i, r in enumerate(pool.imap(worker_simulate, tasks, chunksize=chunk_size)):
            results.append(r)
            if i % max(1, total_tasks // 20) == 0 or i == total_tasks - 1:
                sys.stdout.write(f"\r      Progress: {i+1}/{total_tasks} simulations completed")
                sys.stdout.flush()
                if progress_callback:
                    progress_callback(phase_name, round_idx + 1, len(rounds), i + 1, total_tasks)
        sys.stdout.write("\n")
        
        for i, k in enumerate(active_keys):
            chunk = results[i*run_count : (i+1)*run_count]
            tracker[k]['sum_target'] += sum(r.get(target_metric, 0.0) for r in chunk)
            tracker[k]['sum_floor'] += sum(r.get('highest_floor', 0.0) for r in chunk)
            tracker[k]['runs'] += run_count
            
            # Aggregate specific telemetry for the Dashboard charts
            for r in chunk:
                tracker[k]['floors'].append(int(r.get('highest_floor', 0)))
                for m_k, m_v in r.items():
                    tracker[k]['metrics_sum'][m_k] = tracker[k]['metrics_sum'].get(m_k, 0.0) + m_v
            
        # --- NEW TARGET SORTING LOGIC ---
        def get_sort_key(k):
            if target_metric == 'highest_floor':
                # Sort by the Absolute Max floor reached. If tied, use average floor as a tiebreaker.
                floors = tracker[k]['floors']
                max_f = max(floors) if floors else 0
                avg_f = sum(floors) / len(floors) if floors else 0
                return (max_f, avg_f)
            else:
                score = tracker[k]['sum_target'] / max(1, tracker[k]['runs'])
                floors = tracker[k]['floors']
                avg_f = sum(floors) / len(floors) if floors else 0
                
                # Use Average Floor as a secondary tiebreaker!
                # If 'score' is 0 for all builds, it will correctly pick the build that reached the highest floor.
                return (score, avg_f)
                
        active_keys.sort(key=get_sort_key, reverse=True)
        
        if round_idx < len(rounds) - 1:
            keep_count = max(3, int(len(active_keys) * keep_ratio))
            active_keys = active_keys[:keep_count]

    best_key = active_keys[0]
    best_data = tracker[best_key]
    best_dist = best_data['dist']
    runs_completed = best_data['runs'] if best_data['runs'] > 0 else 1
    
    # Analyze the whole phase to find context for the Confidence Chart
    all_scores = [data['sum_target'] / max(1, data['runs']) for data in tracker.values() if data['runs'] > 0]
    all_scores.sort(reverse=True)
    runner_up = all_scores[1] if len(all_scores) > 1 else all_scores[0]
    worst = all_scores[-1] if all_scores else 0
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Calculate God-Run Probabilities
    floors = best_data['floors']
    abs_max_floor = max(floors) if floors else 0
    abs_max_chance = (floors.count(abs_max_floor) / len(floors)) if floors else 0
    
    # --- SINGLE DETERMINISTIC PROFILING TRACE ---
    p_profile = Player()
    if base_state_dict:
        p_profile.base_stats = base_state_dict.get('base_stats', {}).copy()
        p_profile.asc2_unlocked = base_state_dict.get('asc2_unlocked', False)
        p_profile.arch_level = base_state_dict.get('arch_level', 1)
        p_profile.current_max_floor = base_state_dict.get('current_max_floor', 1)
        p_profile.hades_idol_level = base_state_dict.get('hades_idol_level', 0)
        p_profile.arch_ability_infernal_bonus = base_state_dict.get('arch_ability_infernal_bonus', 0.0)
        p_profile.total_infernal_cards = base_state_dict.get('total_infernal_cards', 0)
        for upg_id, lvl in base_state_dict.get('upgrade_levels', {}).items():
            p_profile.set_upgrade_level(upg_id, lvl)
        for ext_id, lvl in base_state_dict.get('external_levels', {}).items():
            p_profile.set_external_level(ext_id, lvl)
        for card_id, lvl in base_state_dict.get('cards', {}).items():
            p_profile.set_card_level(card_id, lvl)
    else:
        load_state_from_json(p_profile, JSON_PATH)
        
    for k, v in best_dist.items(): p_profile.base_stats[k] = v
    for k, v in fixed_stats.items(): p_profile.base_stats[k] = v
    
    random.seed(42) # Fixed seed ensures a clean, average-looking trace
    sim_profile = CombatSimulator(p_profile)
    sys.stdout = open(os.devnull, 'w')
    profile_state = sim_profile.run_simulation()
    sys.stdout = sys.__stdout__
    
    best_summary = {
        target_metric: best_data['sum_target'] / runs_completed, # Keep for backwards compat
        "avg_floor": best_data['sum_floor'] / runs_completed,
        "abs_max_floor": abs_max_floor,
        "abs_max_chance": abs_max_chance,
        "worst_val": worst,
        "avg_val": avg_score,
        "runner_up_val": runner_up,
        "floors": floors,
        "avg_metrics": {k: v / runs_completed for k, v in best_data['metrics_sum'].items()},
        # Package BOTH arrays so the UI knows exactly what floor/ore we are on
        "stamina_trace": {
            "floor": profile_state.history['floor'],
            "stamina": profile_state.history['stamina']
        }
    }

    if best_summary[target_metric] > 0:
        if target_metric == 'highest_floor':
            print(f"[{phase_name} Winner] {best_dist} -> Peak Floor {abs_max_floor} (Avg: {best_summary['avg_floor']:.1f})")
        else:
            print(f"[{phase_name} Winner] {best_dist} -> {best_summary[target_metric]:,.2f} {target_metric}")
    
    if best_data['runs'] == 0:
        return None, None
        
    return best_dist, best_summary

def benchmark_hardware(baseline_payload, pool, test_iterations=200):
    """Runs a pure baseline micro-benchmark to establish a stable CPU speed."""
    tasks = [baseline_payload for _ in range(test_iterations)]
    
    start_time = time.time()
    pool.map(worker_simulate, tasks)
    elapsed = time.time() - start_time
    
    return test_iterations / elapsed if elapsed > 0 else 1

def get_expected_runs(builds, max_iter):
    """Exactly calculates the number of runs executed through Successive Halving drops."""
    if builds <= 20 or max_iter <= 10: return builds * max_iter
    r1 = max(1, int(max_iter * 0.15))
    r2 = max(1, int(max_iter * 0.35))
    r3 = max_iter - r1 - r2
    runs = builds * r1
    b2 = max(3, int(builds * 0.20))
    runs += b2 * r2
    b3 = max(3, int(b2 * 0.10))
    runs += b3 * r3
    return runs

def get_eta_profiles(stats_list, budget, bounds, sims_per_second, iter_p1=25, iter_p2=50, iter_p3=100):
    """Calculates ETAs dynamically based on exact Halving runs and Positive-Shifted coordinates."""
    profiles = {
        "Fast (Step: 15)": {"step": 15},
        "Standard (Step: 10)": {"step": 10},
        "Deep (Step: 5)": {"step": 5}
    }
    
    # Identify mathematically "free" stats (unlocked)
    free_stats =[s for s in stats_list if bounds[s][0] != bounds[s][1]]
    num_free = len(free_stats)
    
    for name, data in profiles.items():
        step_1 = data["step"]
        step_2 = max(2, step_1 // 3)
        p3_radius = min(2, step_2)
        
        p1_builds = len(generate_distributions(stats_list, budget, step_1, bounds))
        p1_sims = get_expected_runs(p1_builds, iter_p1)
        
        # Positive-Shifted Bounds with Edge-Clipping Factor
        # Real builds hug the 0 and Max Cap walls, cutting the theoretical hypercube search space
        # down by roughly 75%. We generate the full shape, then apply the clipping factor.
        if num_free > 0:
            p2_mock_bounds = {s: (0, 2 * step_1) for s in free_stats}
            p2_budget = ((num_free * step_1) // step_2) * step_2
            raw_p2_builds = len(generate_distributions(free_stats, p2_budget, step_2, p2_mock_bounds))
            
            p3_mock_bounds = {s: (0, 2 * p3_radius) for s in free_stats}
            p3_budget = ((num_free * p3_radius) // 1) * 1
            raw_p3_builds = len(generate_distributions(free_stats, p3_budget, 1, p3_mock_bounds))
            
            EDGE_CLIP = 0.25
            p2_builds = max(1, int(raw_p2_builds * EDGE_CLIP))
            p3_builds = max(1, int(raw_p3_builds * EDGE_CLIP))
        else:
            p2_builds, p3_builds = 0, 0
            
        p2_sims = get_expected_runs(p2_builds, iter_p2)
        p3_sims = get_expected_runs(p3_builds, iter_p3)
        
        total_estimated_builds = p1_builds + p2_builds + p3_builds
        
        # --- SURVIVAL WEIGHTING ---
        weighted_p1 = p1_sims * 0.25 # P1 garbage builds die fast, but still incur parallel overhead
        weighted_p2 = p2_sims * 0.75 # P2 zoomed builds survive much longer
        weighted_p3 = p3_sims * 1.00 # P3 optimized builds take the full benchmark time
        
        total_weighted_sims = weighted_p1 + weighted_p2 + weighted_p3
        
        effective_sims_sec = max(1.0, float(sims_per_second))
        
        estimated_seconds = total_weighted_sims / effective_sims_sec
        
        if estimated_seconds < 60:
            time_str = f"~{int(estimated_seconds)} seconds"
        else:
            time_str = f"~{estimated_seconds/60:.1f} minutes"
            
        data["builds"] = total_estimated_builds
        data["eta_seconds"] = estimated_seconds
        data["time_label"] = time_str
        
    return profiles