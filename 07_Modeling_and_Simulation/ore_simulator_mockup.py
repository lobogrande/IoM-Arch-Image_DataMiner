import os
import json
import random
import pandas as pd

# Define paths to the generated statistics files
SCRIPT_DIR = os.path.dirname(__file__)
SPAWN_RATES_FILE = os.path.join(SCRIPT_DIR, "simulator_spawn_rates.csv")
DROP_TABLES_FILE = os.path.join(SCRIPT_DIR, "simulator_drop_tables.json")

class OreSimulator:
    def __init__(self):
        # Load Gaussian spawn parameters
        try:
            self.spawn_df = pd.read_csv(SPAWN_RATES_FILE)
            if not self.spawn_df.empty:
                self.spawn_df.set_index('floor_id', inplace=True)
                self.spawn_stats = self.spawn_df.to_dict(orient='index')
            else:
                self.spawn_stats = {}
        except FileNotFoundError:
            print(f"Error: {SPAWN_RATES_FILE} not found. Run generate_ore_statistics.py first.")
            self.spawn_stats = {}
            self.spawn_df = pd.DataFrame()

        # Load Drop Tables (Epoch distributions)
        try:
            with open(DROP_TABLES_FILE, 'r') as f:
                self.drop_tables = json.load(f)
        except FileNotFoundError:
            print(f"Error: {DROP_TABLES_FILE} not found. Run generate_ore_statistics.py first.")
            self.drop_tables = {}

        # Global fallbacks just in case a floor isn't in the dataset
        self.global_mean = self.spawn_df['mean_ores'].mean() if not self.spawn_df.empty else 15.5
        self.global_std = self.spawn_df['std_ores'].mean() if not self.spawn_df.empty else 2.5

    def get_epoch_key(self, floor_id):
        """Finds the correct epoch drop table string for the given floor."""
        for epoch_key in self.drop_tables.keys():
            # Parse the "Floors_X_to_Y" string
            parts = epoch_key.replace("Floors_", "").split("_to_")
            start, end = int(parts[0]), int(parts[1])
            if start <= floor_id <= end:
                return epoch_key
        return None

    def populate_floor(self, floor_id):
        """Simulates 24 slots of a floor using calculated probabilities and Gaussian counts."""
        # 1. Get Gaussian parameters for this specific floor (or global fallback)
        if floor_id in self.spawn_stats:
            mu = self.spawn_stats[floor_id]['mean_ores']
            sigma = self.spawn_stats[floor_id]['std_ores']
            min_bound = self.spawn_stats[floor_id]['min_ores']
            max_bound = self.spawn_stats[floor_id]['max_ores']
        else:
            mu, sigma = self.global_mean, self.global_std
            min_bound, max_bound = 8, 24 # Safe fallback limits based on observed game bounds

        # 2. Determine EXACTLY how many ores will spawn on this floor
        if sigma == 0:
            target_ores = int(mu) # Floor is perfectly static in size (e.g., only 1 run of data)
        else:
            # Pick a number from the bell curve, round to nearest integer
            target_ores = round(random.gauss(mu, sigma))
            
            # Ensure the number respects the absolute boundaries seen in the game
            target_ores = max(int(min_bound), min(int(max_bound), target_ores))

        # Absolute grid boundary safety check
        target_ores = max(0, min(24, target_ores))

        # 3. Get the correct Drop Table
        epoch_key = self.get_epoch_key(floor_id)
        if not epoch_key:
            return ["empty"] * 24

        drop_table = self.drop_tables.get(epoch_key, {})
        if not drop_table:
            return ["empty"] * 24

        ore_pool = list(drop_table.keys())
        ore_weights = list(drop_table.values())

        # 4. Generate the required number of random ores based on epoch weights
        if target_ores > 0:
            generated_ores = random.choices(ore_pool, weights=ore_weights, k=target_ores)
        else:
            generated_ores =[]

        # 5. Distribute them randomly across the 24 slots
        grid = ['empty'] * 24
        
        # Pick 'target_ores' unique indices out of the 24 available slots
        chosen_indices = random.sample(range(24), target_ores)
        
        for idx, ore in zip(chosen_indices, generated_ores):
            grid[idx] = ore
            
        return grid

if __name__ == "__main__":
    sim = OreSimulator()
    
    # Quick Test: Let's simulate floor 45 (where leg2 starts, etc.)
    test_floor = 45
    result = sim.populate_floor(test_floor)
    
    print(f"--- Simulated Output for Floor {test_floor} ---")
    
    # Print the 24 slots formatted as a 6x4 grid
    for row in range(4):
        start = row * 6
        end = start + 6
        print([str(x).rjust(6) for x in result[start:end]])