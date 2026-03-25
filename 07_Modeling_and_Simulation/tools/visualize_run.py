# ==============================================================================
# Script: tools/visualize_run.py
# Version: 1.0.0
# Description: Runs a single combat simulation and outputs a graphical dashboard
#              analyzing stamina depletion, speed pool usage, and hit types.
# ==============================================================================

import os
import sys
import matplotlib.pyplot as plt

# Dynamic pathing
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.player import Player
from engine.combat_loop import CombatSimulator
from tools.verify_player import load_state_from_json

def generate_dashboard(state):
    """Takes a RunState object and plots its telemetry data."""
    print("Generating graphical dashboard...")
    
    # Set up a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Run Diagnostics - Final Floor: {state.highest_floor}", fontsize=16, fontweight='bold')

    floors = state.history['floor']

    # --- Plot 1: Stamina Over Time ---
    ax = axs[0, 0]
    ax.plot(floors, state.history['stamina'], color='crimson', linewidth=1.5)
    ax.fill_between(floors, state.history['stamina'], color='crimson', alpha=0.2)
    ax.set_title('Stamina Pool Depletion')
    ax.set_xlabel('Floor Number')
    ax.set_ylabel('Remaining Stamina')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Mark Floor 100 and 150 breakpoints if reached
    if state.highest_floor >= 100:
        ax.axvline(x=100, color='black', linestyle=':', label='Floor 100 Breakpoint')
    if state.highest_floor >= 150:
        ax.axvline(x=150, color='darkorange', linestyle=':', label='Floor 150 Breakpoint')
    ax.legend()

    # --- Plot 2: Speed Mod Pool Over Time ---
    ax = axs[0, 1]
    ax.plot(floors, state.history['speed_pool'], color='dodgerblue', linewidth=1.5)
    ax.fill_between(floors, state.history['speed_pool'], color='dodgerblue', alpha=0.2)
    ax.set_title('Speed Attack Pool Accumulation')
    ax.set_xlabel('Floor Number')
    ax.set_ylabel('Enhanced Attacks Remaining')
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Distribution of Hit Types ---
    ax = axs[1, 0]
    labels =['Normal', 'Crit', 'Super Crit', 'Ultra Crit']
    counts =[state.hit_counts['normal'], state.hit_counts['crit'], 
              state.hit_counts['super'], state.hit_counts['ultra']]
    colors =['#A9A9A9', '#FFD700', '#FF8C00', '#FF00FF'] # Gray, Gold, DarkOrange, Magenta
    
    # Filter out 0 counts for a cleaner pie chart
    filtered_labels =[l for l, c in zip(labels, counts) if c > 0]
    filtered_counts =[c for c in counts if c > 0]
    filtered_colors =[c for l, c in zip(labels, colors) if counts[labels.index(l)] > 0]

    ax.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=140)
    ax.set_title(f"Total Pickaxe Swings: {sum(counts):,}")

    # --- Plot 4: Time Taken per Floor (Pacing) ---
    ax = axs[1, 1]
    # Calculate time spent on each specific floor
    floor_times = {}
    for f, t in zip(floors, state.history['time']):
        if f not in floor_times:
            floor_times[f] = 0
        floor_times[f] = max(floor_times[f], t)
    
    # Derive delta time per floor
    x_floors = sorted(list(floor_times.keys()))
    y_times =[]
    for i in range(len(x_floors)):
        if i == 0:
            y_times.append(floor_times[x_floors[i]])
        else:
            y_times.append(floor_times[x_floors[i]] - floor_times[x_floors[i-1]])
            
    ax.plot(x_floors, y_times, color='mediumseagreen', linewidth=2)
    ax.set_title('Time Spent per Floor (TTK Pacing)')
    ax.set_xlabel('Floor Number')
    ax.set_ylabel('Seconds Spent on Floor')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    p = Player()
    json_path = os.path.join(BASE_DIR, "tools", "player_state.json")
    
    if os.path.exists(json_path):
        print("Loading player state from JSON...")
        load_state_from_json(p, json_path)
    else:
        print("Warning: player_state.json not found. Using baseline stats.")

    # Run the engine
    sim = CombatSimulator(p)
    result_state = sim.run_simulation()
    
    # Show the charts!
    generate_dashboard(result_state)