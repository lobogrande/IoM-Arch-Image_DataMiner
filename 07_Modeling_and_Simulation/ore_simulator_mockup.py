# ==============================================================================
# Script: ore_simulator_mockup.py
# Version: 2.0.0
# Description: Simulates floor generation using Gaussian stats and executes a 
#              serpentine pathing run-through. Calculates damage, handles 4 
#              modifier types, and tracks time-to-clear based on Attack Speed.
# ==============================================================================

import sys
import os
import json
import random
import math
import pandas as pd

# Dynamically link to root-level project_config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

SCRIPT_DIR = os.path.dirname(__file__)
SPAWN_RATES_FILE = os.path.join(SCRIPT_DIR, "simulator_spawn_rates.csv")
DROP_TABLES_FILE = os.path.join(SCRIPT_DIR, "simulator_drop_tables.json")

# ------------------------------------------------------------------------------
# PLAYER STATS & COMBAT FORMULAS
# ------------------------------------------------------------------------------
class Player:
    def __init__(self):
        # BASE STATS (Plug in your formulas / upgrade logic here later)
        self.attack_speed = 2.0  # Attacks per second
        self.base_damage = 5.0   # Base damage before armor
        
        # MODIFIERS:[chance_to_appear (0.0 to 1.0), multiplier_effect]
        # (Rename these as you see fit according to the game mechanics)
        self.mods = {
            'mod1': {'chance': 0.15, 'mult': 2.0},
            'mod2': {'chance': 0.05, 'mult': 5.0},
            'mod3': {'chance': 0.10, 'mult': 1.5},
            'mod4': {'chance': 0.01, 'mult': 10.0}
        }
        
        # RUN TRACKERS
        self.total_time_seconds = 0.0
        self.total_xp = 0
        self.inventory = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} # Fragment Types 1-6

    def calculate_damage_to(self, ore_armor):
        """Formula for calculating actual damage dealt per hit"""
        # TODO: Replace with your exact spreadsheet damage formula
        actual_damage = max(1.0, self.base_damage - ore_armor) 
        return actual_damage

# ------------------------------------------------------------------------------
# ENVIRONMENT & SIMULATION ENGINE
# ------------------------------------------------------------------------------
class OreSimulator:
    def __init__(self):
        # The exact 24-slot indices traversed in Serpentine Order
        # R1 (L->R): 0..5 | R2 (R->L): 11..6 | R3 (L->R): 12..17 | R4 (R->L): 23..18
        self.PATH_ORDER =[
            0, 1, 2, 3, 4, 5, 
            11, 10, 9, 8, 7, 6, 
            12, 13, 14, 15, 16, 17, 
            23, 22, 21, 20, 19, 18
        ]

        # Load Generation Stats
        self._load_generation_data()

    def _load_generation_data(self):
        try:
            self.spawn_df = pd.read_csv(SPAWN_RATES_FILE)
            if not self.spawn_df.empty:
                self.spawn_df.set_index('floor_id', inplace=True)
                self.spawn_stats = self.spawn_df.to_dict(orient='index')
            else:
                self.spawn_stats = {}
        except FileNotFoundError:
            print(f"Error: {SPAWN_RATES_FILE} not found.")
            self.spawn_stats = {}
            self.spawn_df = pd.DataFrame()

        try:
            with open(DROP_TABLES_FILE, 'r') as f:
                self.drop_tables = json.load(f)
        except FileNotFoundError:
            print(f"Error: {DROP_TABLES_FILE} not found.")
            self.drop_tables = {}

        self.global_mean = self.spawn_df['mean_ores'].mean() if not self.spawn_df.empty else 15.5
        self.global_std = self.spawn_df['std_ores'].mean() if not self.spawn_df.empty else 2.5

    def get_epoch_key(self, floor_id):
        for epoch_key in self.drop_tables.keys():
            parts = epoch_key.replace("Floors_", "").split("_to_")
            start, end = int(parts[0]), int(parts[1])
            if start <= floor_id <= end:
                return epoch_key
        return None

    def populate_floor(self, floor_id):
        """Generates the static 24-slot grid based on probability arrays."""
        if floor_id in self.spawn_stats:
            mu = self.spawn_stats[floor_id]['mean_ores']
            sigma = self.spawn_stats[floor_id]['std_ores']
            min_bound = self.spawn_stats[floor_id]['min_ores']
            max_bound = self.spawn_stats[floor_id]['max_ores']
        else:
            mu, sigma = self.global_mean, self.global_std
            min_bound, max_bound = 8, 24 

        if sigma == 0:
            target_ores = int(mu) 
        else:
            target_ores = round(random.gauss(mu, sigma))
            target_ores = max(int(min_bound), min(int(max_bound), target_ores))

        target_ores = max(0, min(24, target_ores))

        epoch_key = self.get_epoch_key(floor_id)
        if not epoch_key:
            return["empty"] * 24

        drop_table = self.drop_tables.get(epoch_key, {})
        if not drop_table:
            return ["empty"] * 24

        ore_pool = list(drop_table.keys())
        ore_weights = list(drop_table.values())

        if target_ores > 0:
            generated_ores = random.choices(ore_pool, weights=ore_weights, k=target_ores)
        else:
            generated_ores =[]

        grid = ['empty'] * 24
        chosen_indices = random.sample(range(24), target_ores)
        for idx, ore in zip(chosen_indices, generated_ores):
            grid[idx] = ore
            
        return grid

    def run_floor(self, floor_id, player: Player):
        """Executes a player's run through a generated floor, tracking time and loot."""
        grid = self.populate_floor(floor_id)
        
        ores_mined = 0
        floor_time = 0.0

        # Run through the slots in the exact serpentine order
        for slot_index in self.PATH_ORDER:
            ore_id = grid[slot_index]
            
            # Movement is instantaneous; we only spend time if there is an ore
            if ore_id == 'empty':
                continue
                
            ores_mined += 1
            ore_stats = cfg.ORE_BASE_STATS.get(ore_id, None)
            
            if not ore_stats:
                print(f"Warning: {ore_id} not found in project_config.py ORE_BASE_STATS. Skipping.")
                continue

            # --- 1. ROLL MODIFIERS ---
            active_multiplier = 1.0
            for mod_name, mod_data in player.mods.items():
                if random.random() <= mod_data['chance']:
                    # TODO: If multiple mods can apply, does it add or multiply? 
                    # Assuming additive multipliers for now:
                    active_multiplier += (mod_data['mult'] - 1.0)

            # --- 2. COMBAT LOGIC ---
            damage_per_hit = player.calculate_damage_to(ore_stats['a'])
            hits_required = math.ceil(ore_stats['hp'] / damage_per_hit)
            
            # Time spent attacking this ore
            time_spent = hits_required / player.attack_speed
            
            # --- 3. REWARD LOGIC ---
            xp_gained = ore_stats['xp'] * active_multiplier
            frag_type = ore_stats['ft']
            frag_amount = ore_stats['fa'] * active_multiplier
            
            # Update Player Tracking
            player.total_time_seconds += time_spent
            floor_time += time_spent
            player.total_xp += xp_gained
            
            if frag_type in player.inventory:
                player.inventory[frag_type] += frag_amount

        return {
            'grid': grid,
            'ores_mined': ores_mined,
            'time_taken': floor_time
        }


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sim = OreSimulator()
    p1 = Player()
    
    test_floor = 45
    print(f"--- Simulating Floor {test_floor} ---")
    
    results = sim.run_floor(test_floor, p1)
    
    # Print the 24 slots formatted as a 6x4 grid so you can visualize it
    print("\n[Generated Grid]")
    for row in range(4):
        start = row * 6
        end = start + 6
        print([str(x).rjust(6) for x in results['grid'][start:end]])

    print(f"\n[Floor Run Results]")
    print(f"Ores Encountered: {results['ores_mined']}")
    print(f"Time Taken to Clear Floor: {results['time_taken']:.2f} seconds")
    
    print("\n[Player Totals]")
    print(f"Total XP: {p1.total_xp:.1f}")
    print(f"Fragments Collected: {p1.inventory}")