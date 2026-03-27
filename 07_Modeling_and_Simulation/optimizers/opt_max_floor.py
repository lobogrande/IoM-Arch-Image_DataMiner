# ==============================================================================
# Script: optimizers/opt_max_floor.py
# Layer 4: Specific Optimizer (Max Floor Push)
# Description: Optimizes stat distribution to reach the highest possible floor
#              before stamina depletion. Focuses on the Absolute Max "God Run".
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
from optimizers.parallel_worker import run_optimization_phase, JSON_PATH, worker_simulate
from project_config import BASE_STAT_CAPS
from collections import Counter

if __name__ == "__main__":
    print("=== AI Arch Optimizer: Maximum Floor Push ===")
    
    p = Player()
    load_state_from_json(p, JSON_PATH)
    
    STATS_TO_OPTIMIZE = ['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
    if p.asc2_unlocked:
        STATS_TO_OPTIMIZE.append('Corr')
        
    DYNAMIC_BUDGET = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
    
    cap_increase = int(p.u('H45'))
    EFFECTIVE_CAPS = {stat: BASE_STAT_CAPS[stat] + cap_increase for stat in STATS_TO_OPTIMIZE}

    print(f"Targeting Metric: highest_floor (Absolute Maximum 'God Run')")
    print(f"Stats in Pool:    {STATS_TO_OPTIMIZE}")
    print(f"Total Budget:     {DYNAMIC_BUDGET} points")
    print(f"Effective Caps:   {EFFECTIVE_CAPS}\n")

    CPU_CORES = max(1, mp.cpu_count() - 1)
    
    # We use progressive scaling, but push Phase 3 heavily to ensure we catch the RNG outliers
    ITER_P1, ITER_P2, ITER_P3 = 25, 50, 100 
    
    start_time = time.time()
    
    # --- RAM INJECTION PAYLOAD ---
    base_state_dict = {
        'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
        'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
        'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
        'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
        'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
        'total_infernal_cards': getattr(p, 'total_infernal_cards', 0)
    }
    
    with mp.Pool(CPU_CORES) as pool:
        target_metric = 'highest_floor'
        
        step_1 = 10
        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
        best_p1, _ = run_optimization_phase(
            "Phase 1 (Coarse)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_1, ITER_P1, pool, FIXED_STATS, bounds_p1,
            base_state_dict=base_state_dict
        )
        
        if not best_p1:
            print(f"\nOptimization Failed: Could not generate valid distributions.")
            sys.exit(1)
            
        bounds_p2 = {}
        for stat in STATS_TO_OPTIMIZE:
            val = best_p1[stat]
            bounds_p2[stat] = (max(0, val - step_1), min(EFFECTIVE_CAPS[stat], val + step_1))
            
        step_2 = 3
        best_p2, _ = run_optimization_phase(
            "Phase 2 (Fine)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, step_2, ITER_P2, pool, FIXED_STATS, bounds_p2,
            base_state_dict=base_state_dict
        )
        
        bounds_p3 = {}
        if best_p2:
            p3_radius = min(2, step_2)
            for stat in STATS_TO_OPTIMIZE:
                val = best_p2[stat]
                bounds_p3[stat] = (max(0, val - p3_radius), min(EFFECTIVE_CAPS[stat], val + p3_radius))
        else:
            bounds_p3 = bounds_p2

        best_p3, _ = run_optimization_phase(
            "Phase 3 (Exact)", target_metric, STATS_TO_OPTIMIZE, 
            DYNAMIC_BUDGET, 1, ITER_P3, pool, FIXED_STATS, bounds_p3,
            base_state_dict=base_state_dict
        )
        
        # ======================================================================
        # PHASE 4: PEAK-MUTATION SYNTHESIS (GRADIENT POLISH)
        # ======================================================================
        print("\n[Phase 4 (Synthesis)] Mapping neighborhood and running deep 200-sim verification...")
        
        seed_dist = best_p3 or best_p2 or best_p1
        neighbors = [seed_dist]
        
        # Map all valid 1-point swaps around the seed
        for s_from in STATS_TO_OPTIMIZE:
            if seed_dist[s_from] > 0:
                for s_to in STATS_TO_OPTIMIZE:
                    if s_from != s_to and seed_dist[s_to] < EFFECTIVE_CAPS[s_to]:
                        neighbor = seed_dist.copy()
                        neighbor[s_from] -= 1
                        neighbor[s_to] += 1
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
                            
        verify_args =[]
        for build in neighbors:
            for _ in range(200):
                verify_args.append({'stats': build, 'fixed_stats': FIXED_STATS, 'state_dict': base_state_dict, '_b_id': tuple(build.items())})
                
        res_list = pool.map(worker_simulate, verify_args)
        
        build_results = {}
        for args, r in zip(verify_args, res_list):
            b_id = args['_b_id']
            if b_id not in build_results:
                build_results[b_id] = {'sum_t': 0, 'sum_f': 0, 'floors': []}
            build_results[b_id]['sum_t'] += r.get(target_metric, r.get("highest_floor", 0))
            build_results[b_id]['sum_f'] += r.get("highest_floor", 0)
            build_results[b_id]['floors'].append(r.get("highest_floor", 0))
            
        def sort_key(b_id):
            data = build_results[b_id]
            return (max(data['floors']), data['sum_f']) # Sort strictly by Absolute Max Floor, then Average Floor
            
        best_b_id = sorted(build_results.keys(), key=sort_key, reverse=True)[0]
        final_meta_data = build_results[best_b_id]
        final_meta_dist = dict(best_b_id)
        
    elapsed = time.time() - start_time
    
    # --- UX: Final Output Readout ---
    abs_max = max(final_meta_data['floors'])
    abs_chance = (final_meta_data['floors'].count(abs_max) / 200.0) * 100
    avg_floor = final_meta_data['sum_f'] / 200.0
    
    print("\n" + "="*50)
    print(f"Optimization & Synthesis Complete in {elapsed:.2f} seconds.")
    print("🧬 Synthesized Meta-Build (Peak Variance):")
    for stat in STATS_TO_OPTIMIZE:
        change_str = ""
        if final_meta_dist[stat] != seed_dist[stat]:
            diff = final_meta_dist[stat] - seed_dist[stat]
            change_str = f" ({'+' if diff > 0 else ''}{diff} from Phase 3)"
        print(f"  {stat}: {final_meta_dist[stat]}{change_str}")
        
    for k, v in FIXED_STATS.items():
        print(f"  {k}: {v} (Fixed)")
        
    print(f"\n[ PERFORMANCE PROOF (200 SIMULATIONS) ]")
    print(f" - Absolute Maximum Floor Reached: Floor {abs_max}!!!")
    print(f" - Likelihood of God-Run:          {abs_chance:.1f}% ({final_meta_data['floors'].count(abs_max)} in 200 runs)")
    print(f" - Average Consistency Floor:      Floor {avg_floor:,.1f}")
    print("="*50)