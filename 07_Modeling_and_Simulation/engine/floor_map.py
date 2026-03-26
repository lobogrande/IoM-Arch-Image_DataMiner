# ==============================================================================
# Script: engine/floor_map.py
# Version: 1.0.1 (Modular Architecture)
# Description: Generates the 24-slot environment. Handles the Gaussian spawn 
#              math, instantiates Ore objects, and rolls RNG for Gleaming 
#              Floors and individual Ore Modifiers.
# ==============================================================================

import os
import sys
import json
import random
import pandas as pd

# --- BULLETPROOF PATHING ---
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

ROOT_DIR = os.path.abspath(os.path.join(SIM_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ore import Ore

SPAWN_RATES_FILE = os.path.join(SIM_DIR, "simulator_spawn_rates.csv")
DROP_TABLES_FILE = os.path.join(SIM_DIR, "simulator_drop_tables.json")


class Floor:
    """
    Represents a single generated floor instance containing a 24-slot grid.
    """
    def __init__(self, floor_id, grid, is_gleaming, gleaming_multi):
        self.floor_id = floor_id
        self.grid = grid  # Array of 24 slots (either an Ore object or None)
        self.is_gleaming = is_gleaming
        self.gleaming_multi = gleaming_multi


class FloorGenerator:
    """
    Loads Gaussian spawn data into memory once and rapidly generates Floor objects.
    """
    def __init__(self):
        self._load_data()

    def _load_data(self):
        """Loads the CSV and JSON probability maps into memory."""
        # Load Gaussian spawn parameters
        try:
            self.spawn_df = pd.read_csv(SPAWN_RATES_FILE)
            if not self.spawn_df.empty:
                self.spawn_df.set_index('floor_id', inplace=True)
                self.spawn_stats = self.spawn_df.to_dict(orient='index')
            else:
                self.spawn_stats = {}
        except FileNotFoundError:
            print(f"[ERROR] {SPAWN_RATES_FILE} not found. Ensure it is in the root simulator folder.")
            self.spawn_stats = {}
            self.spawn_df = pd.DataFrame()

        # Load Drop Tables (Epoch distributions)
        try:
            with open(DROP_TABLES_FILE, 'r') as f:
                self.drop_tables = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] {DROP_TABLES_FILE} not found. Ensure it is in the root simulator folder.")
            self.drop_tables = {}

        # Global fallbacks just in case a simulated floor isn't in the historic dataset
        self.global_mean = self.spawn_df['mean_ores'].mean() if not self.spawn_df.empty else 15.5
        self.global_std = self.spawn_df['std_ores'].mean() if not self.spawn_df.empty else 2.5

    def get_epoch_key(self, floor_id):
        """Finds the correct epoch drop table string for the given floor."""
        for epoch_key in self.drop_tables.keys():
            parts = epoch_key.replace("Floors_", "").split("_to_")
            start, end = int(parts[0]), int(parts[1])
            if start <= floor_id <= end:
                return epoch_key
        return None

    def generate_floor(self, floor_id, player):
        """
        Calculates spawn volume, rolls for modifiers, and returns a fully 
        populated 24-slot Floor object.
        """
        # 1. Roll for Gleaming Floor
        is_gleaming = random.random() < player.gleaming_floor_chance
        gleaming_multi = player.gleaming_floor_multi if is_gleaming else 1.0

        # 2. Get Gaussian parameters for this specific floor
        if floor_id in self.spawn_stats:
            mu = self.spawn_stats[floor_id]['mean_ores']
            sigma = self.spawn_stats[floor_id]['std_ores']
            min_bound = self.spawn_stats[floor_id]['min_ores']
            max_bound = self.spawn_stats[floor_id]['max_ores']
        else:
            mu, sigma = self.global_mean, self.global_std
            min_bound, max_bound = 8, 24 

        # 3. Determine EXACTLY how many ores will spawn
        if sigma == 0:
            target_ores = int(mu) 
        else:
            target_ores = round(random.gauss(mu, sigma))
            target_ores = max(int(min_bound), min(int(max_bound), target_ores))

        # Final absolute bounds safety check
        target_ores = max(0, min(24, target_ores))

        # 4. Get the correct Drop Table
        epoch_key = self.get_epoch_key(floor_id)
        drop_table = self.drop_tables.get(epoch_key, {}) if epoch_key else {}

        if not drop_table or target_ores == 0:
            return Floor(floor_id, [None] * 24, is_gleaming, gleaming_multi)

        # --- EXCEL-BASED DIV3 OVERRIDE (The "Breather" Mechanic) ---
        div3_count = 0
        if floor_id > 99:
            r = random.random()
            if r < 0.45: div3_count = 0
            elif r < 0.85: div3_count = 1
            elif r < 0.98: div3_count = 2
            elif r < 0.995: div3_count = 3
            else: div3_count = 4
            
        # Ensure we don't accidentally spawn more Div3s than total ores allowed
        div3_count = min(div3_count, target_ores)
        remaining_target = target_ores - div3_count

        # Filter the generic drop table so we don't double-roll Div3 ores
        ore_pool =[]
        ore_weights =[]
        for ore_name, weight in drop_table.items():
            if floor_id > 99 and ore_name == 'div3':
                continue
            ore_pool.append(ore_name)
            ore_weights.append(weight)

        generated_ore_ids =[]
        
        # Roll the remaining non-Div3 ores based on standard probability
        if remaining_target > 0 and ore_pool:
            generated_ore_ids = random.choices(ore_pool, weights=ore_weights, k=remaining_target)

        # Append our explicitly controlled Div3 ores
        generated_ore_ids.extend(['div3'] * div3_count)

        # 5. Distribute them into the 24 slots randomly
        grid = [None] * 24
        chosen_indices = random.sample(range(24), target_ores)
        
        for idx, ore_id in zip(chosen_indices, generated_ore_ids):
            # Instantiate the Ore object
            ore = Ore(ore_id, floor_id, player)
            
            # Roll for Random Modifiers
            ore.modifiers = {
                'exp_multi': player.exp_mod_gain if (random.random() < player.exp_mod_chance) else 1.0,
                'loot_multi': player.loot_mod_gain if (random.random() < player.loot_mod_chance) else 1.0,
                'stamina_gain': player.stamina_mod_gain if (random.random() < player.stamina_mod_chance) else 0.0,
                
                # Speed Mod grants a flat pool of attacks upon death
                'speed_active': random.random() < player.speed_mod_chance,
                'speed_gain': player.speed_mod_gain
            }
            
            # Place in grid
            grid[idx] = ore
            
        return Floor(floor_id, grid, is_gleaming, gleaming_multi)

# ==============================================================================
# QUICK VERIFICATION TEST
# ==============================================================================
if __name__ == "__main__":
    from core.player import Player
    
    # Setup Player
    p = Player()
    p.base_stats['Luck'] = 100 # Boost luck to force some modifiers to appear
    p.set_upgrade_level(38, 20) # Boost Exp Mod Gain/Chance
    p.asc2_unlocked = False

    # Initialize Generator
    generator = FloorGenerator()
    
    test_floor_id = 45
    print(f"--- Generating Floor {test_floor_id} ---")
    
    # Generate the floor
    floor = generator.generate_floor(test_floor_id, p)
    
    print(f"Gleaming Floor: {floor.is_gleaming} (Multi: {floor.gleaming_multi}x)")
    print(f"Ores Spawned: {sum(1 for slot in floor.grid if slot is not None)}/24")
    
    print("\n[ Modifier Check on Spawned Ores ]")
    for i, slot in enumerate(floor.grid):
        if slot is not None:
            mods = slot.modifiers
            # Only print if an ore rolled a modifier
            if mods['exp_multi'] > 1.0 or mods['loot_multi'] > 1.0 or mods['stamina_gain'] > 0 or mods['speed_active']:
                print(f"Slot {i} ({slot.ore_id}): Exp {mods['exp_multi']}x | Loot {mods['loot_multi']}x | Sta +{mods['stamina_gain']} | SpdActive: {mods['speed_active']}")