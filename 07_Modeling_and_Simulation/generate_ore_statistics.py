# ==============================================================================
# Script: generate_ore_statistics.py
# Version: 1.1.0
# Description: Ingests run data to calculate Gaussian spawn rates and epoch-based 
#              ore drop probabilities. Excludes boss floors.
# ==============================================================================

import sys
import os
import glob
import json
import pandas as pd
import numpy as np

# Dynamically link to root-level project_config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
# Target the correct tracking archive folder based on project structure
DATA_PATH = os.path.join("..", "Data_02_Tracking_Archive", "floor_ore_inventory_run_*.csv")
OUTPUT_DIR = os.path.dirname(__file__)

# Generate the 24 slot column names (R1_S0 to R4_S5)
SLOT_COLS =[f"R{r}_S{s}" for r in range(1, 5) for s in range(6)]

def calculate_epochs(restrictions):
    """Calculates distinct floor ranges (epochs) where the ore pool changes."""
    boundaries = {1}
    for min_f, max_f in restrictions.values():
        boundaries.add(min_f)
        if max_f != 999:
            boundaries.add(max_f + 1)
            
    boundaries = sorted(list(boundaries))
    epochs =[]
    for i in range(len(boundaries) - 1):
        epochs.append((boundaries[i], boundaries[i+1] - 1))
    
    epochs.append((boundaries[-1], 999))
    return epochs

def main():
    print("Loading data and calculating epochs...")
    epochs = calculate_epochs(cfg.ORE_RESTRICTIONS)
    
    file_list = glob.glob(DATA_PATH)
    if not file_list:
        print(f"Warning: No CSV files found matching {DATA_PATH}")
        return

    all_data =[]
    for file in file_list:
        df = pd.read_csv(file)
        # Verify columns exist, extract just floor_id and slots
        available_cols = ['floor_id'] +[col for col in SLOT_COLS if col in df.columns]
        df = df[available_cols]
        all_data.append(df)

    master_df = pd.concat(all_data, ignore_index=True)

    # --- REMOVE BOSS FLOORS ---
    boss_floors = set(cfg.BOSS_DATA.keys())
    master_df = master_df[~master_df['floor_id'].isin(boss_floors)]

    # Fill NaNs in slot columns with 'empty'
    master_df[SLOT_COLS] = master_df[SLOT_COLS].fillna('empty')

    # --- PROCESS SPAWN RATES (GAUSSIAN MODEL) ---
    print("Calculating Gaussian spawn models...")
    # Calculate the total number of ores that spawned on every single row (floor run)
    master_df['total_ores'] = master_df[SLOT_COLS].apply(lambda x: (x != 'empty').sum(), axis=1)

    # Group by floor to get the Mean and Standard Deviation of total ores
    spawn_rates = master_df.groupby('floor_id')['total_ores'].agg(
        mean_ores='mean',
        std_ores='std',
        min_ores='min',
        max_ores='max'
    ).reset_index()

    # Fill NaN standard deviations (happens if a floor only has 1 run in the dataset) with 0
    spawn_rates['std_ores'] = spawn_rates['std_ores'].fillna(0)

    spawn_csv_path = os.path.join(OUTPUT_DIR, "simulator_spawn_rates.csv")
    spawn_rates.to_csv(spawn_csv_path, index=False)
    print(f"Saved Gaussian spawn rates to {spawn_csv_path}")

    # --- PROCESS DROP TABLES PER EPOCH ---
    print("Calculating drop tables per epoch...")
    melted_df = master_df.melt(id_vars=['floor_id'], value_vars=SLOT_COLS, 
                               var_name='slot', value_name='ore_id')
    
    ores_only_df = melted_df[melted_df['ore_id'] != 'empty'].copy()

    def get_epoch_label(floor):
        for start, end in epochs:
            if start <= floor <= end:
                return f"Floors_{start}_to_{end}"
        return "Unknown"

    ores_only_df['epoch'] = ores_only_df['floor_id'].apply(get_epoch_label)

    epoch_stats = ores_only_df.groupby(['epoch', 'ore_id']).size().reset_index(name='count')
    epoch_totals = epoch_stats.groupby('epoch')['count'].transform('sum')
    epoch_stats['probability_weight'] = epoch_stats['count'] / epoch_totals

    dist_matrix = epoch_stats.pivot(index='epoch', columns='ore_id', values='probability_weight').fillna(0)
    
    # Sort epochs chronologically
    dist_matrix['sort_key'] = dist_matrix.index.str.extract(r'Floors_(\d+)_').astype(int)
    dist_matrix = dist_matrix.sort_values('sort_key').drop(columns=['sort_key'])

    dist_csv_path = os.path.join(OUTPUT_DIR, "simulator_ore_distributions.csv")
    dist_matrix.to_csv(dist_csv_path)
    print(f"Saved ore distributions to {dist_csv_path}")

    # --- GENERATE JSON FOR SIMULATOR ---
    simulator_config = {}
    for index, row in dist_matrix.iterrows():
        weights = row[row > 0].to_dict()
        simulator_config[index] = weights

    json_path = os.path.join(OUTPUT_DIR, "simulator_drop_tables.json")
    with open(json_path, "w") as f:
        json.dump(simulator_config, f, indent=4)
    print(f"Saved JSON configuration to {json_path}")

if __name__ == "__main__":
    main()