# ==============================================================================
# Script: stress_test.py
# Description: Brute-forces a single stat distribution 10,000 times to find 
#              the absolute mathematical ceiling (The 99.99th percentile God Run).
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp
from collections import Counter

# Set up paths so the workers can find the simulator modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tools.verify_player import load_state_from_json
from core.player import Player
from engine.combat_loop import CombatSimulator
from optimizers.parallel_worker import JSON_PATH

def worker_stress(payload):
    """Custom worker that runs a pure simulation."""
    p = Player()
    if os.path.exists(JSON_PATH):
        load_state_from_json(p, JSON_PATH)
    
    # Apply the stats exactly as passed by the main thread
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    
    # Silence the simulation print statements so the console doesn't flood
    sys.stdout = open(os.devnull, 'w')
    result = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    return result.highest_floor

if __name__ == "__main__":
    print("=== AI Arch Optimizer: 10,000x God-Run Stress Test ===")
    
    # 1. Load the player dynamically from the JSON file
    p_main = Player()
    if os.path.exists(JSON_PATH):
        load_state_from_json(p_main, JSON_PATH)
        print(f"Loaded player data from: {JSON_PATH}")
    else:
        print(f"[ERROR] Could not find {JSON_PATH}. Make sure it is in the tools folder!")
        sys.exit(1)
        
    # 2. Extract their exact Base Stats to pass to the workers
    payload = {
        'stats': p_main.base_stats
    }
    
    TEST_RUNS = 10000
    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    print(f"Stats Locked: {payload['stats']}")
    print(f"Simulating {TEST_RUNS:,} distinct lives across {CPU_CORES} CPU cores...\n")
    
    tasks = [payload for _ in range(TEST_RUNS)]
    
    start_time = time.time()
    results =[]
    
    # Run the multiprocessing pool
    with mp.Pool(CPU_CORES) as pool:
        for i, floor in enumerate(pool.imap_unordered(worker_stress, tasks, chunksize=100)):
            results.append(floor)
            if (i + 1) % 1000 == 0:
                print(f"  -> {i + 1:,} / {TEST_RUNS:,} lives simulated...")

    elapsed = time.time() - start_time
    max_floor = max(results)
    avg_floor = sum(results) / len(results)
    
    counts = Counter(results)
    
    print("\n" + "="*50)
    print(f"Stress Test Complete in {elapsed:.1f} seconds.")
    print(f"Average Consistency Floor: {avg_floor:.1f}")
    print(f"ABSOLUTE CEILING REACHED:  Floor {max_floor}!!!")
    print("-" * 50)
    print("Top 3 God-Runs:")
    for floor in sorted(counts.keys(), reverse=True)[:3]:
        chance = (counts[floor] / TEST_RUNS) * 100
        print(f"  Floor {floor}: Hit {counts[floor]:,} times ({chance:.3f}%)")
    print("="*50)