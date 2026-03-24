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
        # Load spawn densities
        try:
            self.spawn_df = pd.read_csv(SPAWN_RATES_FILE)
            # Create a dictionary of {floor_id: chance_to_spawn}
            self.spawn_rates = dict(zip(self.spawn_df.floor_id, self.spawn_df.chance_to_spawn))
        except FileNotFoundError:
            print("Error: Run generate_ore_statistics.py first to create spawn rates.")
            self.spawn_rates = {}

        # Load Drop Tables (Epoch distributions)
        try:
            with open(DROP_TABLES_FILE, 'r') as f:
                self.drop_tables = json.load(f)
        except FileNotFoundError:
            print("Error: Run generate_ore_statistics.py first to create drop tables.")
            self.drop_tables = {}

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
        """Simulates 24 slots of a floor using calculated probabilities."""
        # 1. Determine density (default to a global average if floor not in dataset)
        spawn_chance = self.spawn_rates.get(floor_id, 0.35) # example fallback: 35%
        
        # 2. Grab the appropriate probability table
        epoch_key = self.get_epoch_key(floor_id)
        if not epoch_key:
            return ["empty"] * 24

        drop_table = self.drop_tables[epoch_key]
        ore_pool = list(drop_table.keys())
        ore_weights = list(drop_table.values())

        # 3. Roll for 24 slots
        grid =[]
        for _ in range(24):
            if random.random() < spawn_chance:
                # Pick an ore based on its weighted probability
                chosen_ore = random.choices(ore_pool, weights=ore_weights, k=1)[0]
                grid.append(chosen_ore)
            else:
                grid.append('empty')
        
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