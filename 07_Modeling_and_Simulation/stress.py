# ==============================================================================
# Script: stress_test.py
# Description: Parameter Sweep for hidden Crosshair Spawn Intervals. 
#              Hunts for the specific interval that allows the engine to hit 
#              Floor 139 using the player's 30-40% Auto-Tap chance.
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tools.verify_player import load_state_from_json
from core.player import Player
from engine.combat_loop import CombatSimulator
from optimizers.parallel_worker import JSON_PATH

def worker_sweep(payload):
    """Custom worker that accepts a dynamic crosshair interval."""
    p = Player()
    if os.path.exists(JSON_PATH):
        load_state_from_json(p, JSON_PATH)
    
    for stat_name, val in payload['stats'].items():
        p.base_stats[stat_name] = val
    for stat_name, val in payload['fixed_stats'].items():
        p.base_stats[stat_name] = val
        
    sim = CombatSimulator(p)
    # Inject the interval dynamically into the engine
    sim.crosshair_interval = payload['interval']
    
    # Silence the simulation print statements so the console doesn't flood
    sys.stdout = open(os.devnull, 'w')
    result = sim.run_simulation()
    sys.stdout = sys.__stdout__
    
    return result.highest_floor

if __name__ == "__main__":
    print("=== AI Arch Optimizer: Crosshair Parameter Sweep ===")
    
    payload = {
        'stats': {'Str': 45, 'Agi': 10, 'Per': 3, 'Int': 0, 'Luck': 22, 'Div': 15},
        'fixed_stats': {'Corr': 0}
    }
    
    # We will test a crosshair spawning every X seconds
    INTERVALS_TO_TEST =[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    TEST_RUNS_PER_INTERVAL = 2000
    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    print(f"Stats Locked: {payload['stats']}")
    print(f"Hunting for Floor 139 across {len(INTERVALS_TO_TEST)} intervals...\n")
    print("-" * 50)
    
    with mp.Pool(CPU_CORES) as pool:
        for interval in INTERVALS_TO_TEST:
            start_time = time.time()
            print(f"Testing Spawn Rate: 1 Crosshair every {interval}s... ", end="", flush=True)
            
            tasks = [{'stats': payload['stats'], 'fixed_stats': payload['fixed_stats'], 'interval': interval} for _ in range(TEST_RUNS_PER_INTERVAL)]
            
            # Executing the pool map
            results = pool.map(worker_sweep, tasks)
            
            elapsed = time.time() - start_time
            max_floor = max(results)
            avg_floor = sum(results) / len(results)
            
            # Count the occurrences of the top floor
            counts = Counter(results)
            top_floor_count = counts[max_floor]
            chance = (top_floor_count / TEST_RUNS_PER_INTERVAL) * 100
            
            print(f"Done in {elapsed:.1f}s")
            print(f"  -> Absolute Ceiling: Floor {max_floor} (Hit {top_floor_count} times, {chance:.2f}%)")
            print(f"  -> Average Floor:    {avg_floor:.1f}")
            print("-" * 50)
            
    print("\nSweep Complete!")