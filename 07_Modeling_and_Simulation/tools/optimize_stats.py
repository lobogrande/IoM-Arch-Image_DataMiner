# ==============================================================================
# Script: tools/optimize_stats.py
# Layer 4: Monte Carlo Optimizer
# Description: Uses multiprocessing to simulate thousands of combat runs across
#              different stat distributions to find the mathematical optimum.
# ==============================================================================

import os
import sys
import random
import multiprocessing as mp
import time
from itertools import combinations_with_replacement

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from engine.combat_loop import CombatSimulator
from tools.verify_player import load_state_from_json

JSON_PATH = os.path.join(BASE_DIR, "tools", "player_state.json")

def worker_simulate(stat_overrides):
    """
    Top-level worker function for multiprocessing. 
    Re-instantiates the Player and Simulator to avoid pickling limits and RNG locking.
    """
    # 1. Force a new RNG seed for this specific process
    random.seed(os.urandom(4))
    
    # 2. Build the baseline player
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    # 3. Apply the stat mutations we want to test
    for stat_name, val in stat_overrides.items():
        p.base_stats[stat_name] = val
        
    # 4. Run the simulation
    sim = CombatSimulator(p)
    
    # Temporarily suppress print statements from the loop for cleaner terminal output
    sys.stdout = open(os.devnull, 'w')
    result_state = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    # 5. Calculate Return Metrics (Primitives only to ensure safe pickling)
    runtime_mins = result_state.total_time / 60.0
    xp_per_min = result_state.total_xp / runtime_mins if runtime_mins > 0 else 0
    
    return {
        "highest_floor": result_state.highest_floor,
        "total_xp": result_state.total_xp,
        "runtime_mins": runtime_mins,
        "xp_per_min": xp_per_min,
        "ores_mined": result_state.ores_mined
    }

def generate_stat_distributions(total_points, step=10):
    """
    Generates different stat combinations. 
    For speed, we will only mutate 3 main stats: Str, Agi, Luck.
    You can expand this to Per, Int, Div later.
    """
    distributions =[]
    # Find all ways to distribute `total_points` into 3 buckets using `step` increments
    for str_pts in range(0, total_points + 1, step):
        for agi_pts in range(0, total_points - str_pts + 1, step):
            luck_pts = total_points - str_pts - agi_pts
            distributions.append({
                'Str': str_pts,
                'Agi': agi_pts,
                'Luck': luck_pts,
                # Keep others static for this test
                'Per': 0, 'Int': 0, 'Div': 15, 'Corr': 0
            })
    return distributions

def evaluate_distribution(dist, iterations_per_dist, pool):
    """Runs a specific distribution N times and averages the results."""
    # Create a list of N identical tasks for the pool
    tasks =[dist for _ in range(iterations_per_dist)]
    
    results = pool.map(worker_simulate, tasks)
    
    # Aggregate results
    avg_xp_min = sum(r['xp_per_min'] for r in results) / iterations_per_dist
    avg_floor = sum(r['highest_floor'] for r in results) / iterations_per_dist
    avg_runtime = sum(r['runtime_mins'] for r in results) / iterations_per_dist
    
    return {
        "distribution": dist,
        "avg_xp_min": avg_xp_min,
        "avg_floor": avg_floor,
        "avg_runtime": avg_runtime
    }

if __name__ == "__main__":
    print("=== AI Arch Monte Carlo Optimizer ===")
    
    # Configuration
    TOTAL_STAT_POINTS = 80 # (50 Str + 0 Agi + 30 Luck from your JSON)
    ITERATIONS_PER_DIST = 100 # How many runs per stat variation
    CPU_CORES = max(1, mp.cpu_count() - 1) # Leave 1 core free for OS
    
    print(f"Generating stat distributions (Budget: {TOTAL_STAT_POINTS} pts)...")
    distributions_to_test = generate_stat_distributions(TOTAL_STAT_POINTS, step=20)
    print(f"Testing {len(distributions_to_test)} different builds.")
    print(f"Running {ITERATIONS_PER_DIST} simulations per build on {CPU_CORES} CPU cores...\n")
    
    start_time = time.time()
    
    best_xp_build = None
    best_xp_val = 0.0
    
    # Initialize Multiprocessing Pool
    with mp.Pool(CPU_CORES) as pool:
        for i, dist in enumerate(distributions_to_test):
            summary = evaluate_distribution(dist, ITERATIONS_PER_DIST, pool)
            
            str_val = dist['Str']
            agi_val = dist['Agi']
            luck_val = dist['Luck']
            
            print(f"[{i+1}/{len(distributions_to_test)}] Str:{str_val:2d} | Agi:{agi_val:2d} | Luck:{luck_val:2d}  -->  "
                  f"Avg Floor: {summary['avg_floor']:.1f} | Avg Runtime: {summary['avg_runtime']:.1f}m | "
                  f"Avg XP/Min: {summary['avg_xp_min']:,.0f}")
            
            if summary['avg_xp_min'] > best_xp_val:
                best_xp_val = summary['avg_xp_min']
                best_xp_build = summary
                
    elapsed = time.time() - start_time
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print("\nBEST BUILD FOR XP/MINUTE:")
    print(f"Stats:   {best_xp_build['distribution']}")
    print(f"XP/Min:  {best_xp_build['avg_xp_min']:,.0f}")
    print(f"Avg Flr: {best_xp_build['avg_floor']:.1f}")