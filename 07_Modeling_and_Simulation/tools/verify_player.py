# ==============================================================================
# Script: tools/verify_player.py
# Version: 1.0.0 (Modular Architecture)
# Description: A standalone CLI tool to auto-generate JSON state templates, 
#              load them into the Player engine, and print verification reports.
# ==============================================================================

import sys
import os
import json

# Dynamically add the parent directory to Python's path so we can import from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.player import Player

SCRIPT_DIR = os.path.dirname(__file__)

def generate_json_template(player_class, filepath):
    """Generates a blank JSON file based on the Player class definitions."""
    template = {
        "settings": {
            "asc2_unlocked": False, "arch_level": 1, "current_max_floor": 100,
            "base_damage_const": 10, "hades_idol_level": 0, "total_infernal_cards": 0,
            "arch_ability_infernal_bonus": 0.0
        },
        "base_stats": {"Str": 0, "Agi": 0, "Per": 0, "Int": 0, "Luck": 0, "Div": 0, "Corr": 0},
        "internal_upgrades": {str(k): 0 for k in player_class.UPGRADE_DEF.keys()},
        "external_upgrades": {str(k): 0 for k in player_class.EXTERNAL_DEF.keys()},
        # Temporary instance just to grab the default keys
        "cards": {k: 0 for k in player_class().cards.keys()} 
    }
    with open(filepath, 'w') as f: 
        json.dump(template, f, indent=4)

def load_state_from_json(player, filepath):
    """Loads a user's JSON file and populates the Player instance."""
    with open(filepath, 'r') as f: 
        data = json.load(f)
        
    s = data.get('settings', {})
    player.asc2_unlocked = s.get('asc2_unlocked', False)
    player.arch_level = s.get('arch_level', 1)
    player.current_max_floor = s.get('current_max_floor', 100)
    player.base_damage_const = s.get('base_damage_const', 10)
    player.hades_idol_level = s.get('hades_idol_level', 0)
    player.total_infernal_cards = s.get('total_infernal_cards', 0)
    player.arch_ability_infernal_bonus = s.get('arch_ability_infernal_bonus', 0.0)
    
    for stat, val in data.get('base_stats', {}).items(): 
        player.base_stats[stat] = val
        
    for row, lvl in data.get('internal_upgrades', {}).items(): 
        player.set_upgrade_level(int(row), lvl)
        
    for row, lvl in data.get('external_upgrades', {}).items(): 
        player.set_external_level(int(row), lvl)
        
    for ore_id, lvl in data.get('cards', {}).items(): 
        player.set_card_level(ore_id, lvl)

def print_debug_equations(player):
    """Prints the raw mathematical traces for easy spreadsheet verification."""
    print("\n" + "="*50)
    print(" 🔍 DEBUG EQUATION TRACES 🔍")
    print("="*50)
    
    # Max Stamina Trace
    f14, f23, h39, f3 = player.u('F14'), player.u('F23'), player.u('H39'), player.u('F3')
    stat_agi, f26 = player.stat('Agi'), player.u('F26')
    h28, f54, corr = player.u('H28'), player.u('F54'), player.stat('Corr')
    
    base = 100 + f14 + f23 + h39 + f3
    agi_comp = stat_agi * (5 + f26)
    asc2_comp = (1 + h28 + f54) * (1 - 0.03 * corr)
    floor_comp = 1 + (0.01 * min(100, player.current_max_floor))
    
    print(f"\nMax Stamina Breakdown:")
    print(f"  Base (100+F14+F23+H39+F3): 100 + {f14} + {f23} + {h39} + {f3} = {base}")
    print(f"  Agi Comp (M4*(5+F26)):     {stat_agi} * (5 + {f26}) = {agi_comp}")
    print(f"  Asc2 Comp:                 (1 + {h28} + {f54}) * (1 - 0.03*{corr}) = {asc2_comp:.4f}")
    print(f"  Floor Comp:                1 + 0.01*{min(100, player.current_max_floor)} = {floor_comp:.2f}")
    print(f"  Calculation:               ({base} + {agi_comp}) * {asc2_comp:.4f} * {floor_comp:.2f} = {(base+agi_comp)*asc2_comp*floor_comp:.2f}")
    print("="*50 + "\n")

def print_character_sheet(player):
    """Prints a highly formatted layout of all calculated stats to terminal."""
    print("\n" + "="*50)
    print(" 🎮 PLAYER STATE VERIFICATION REPORT 🎮")
    print("="*50)
    
    print(f"\n[ COMBAT STATS ]")
    print(f"Max Stamina:       {player.max_sta:,}")
    print(f"Damage:            {player.damage:,}")
    print(f"Armor Pen:         {player.armor_pen:,}")
    print(f"Attack Speed:      {player.atk_spd:.2f}")

    print(f"\n[ CRITICAL SYSTEM ]")
    print(f"Crit Chance:       {player.crit_chance*100:.2f}%")
    print(f"Crit Dmg Mult:     {player.crit_dmg_mult:.2f}x")
    print(f"S. Crit Chance:    {player.super_crit_chance*100:.2f}%")
    print(f"S. Crit Dmg Mult:  {player.super_crit_dmg_mult:.2f}x")
    print(f"U. Crit Chance:    {player.ultra_crit_chance*100:.2f}%")
    print(f"U. Crit Dmg Mult:  {player.ultra_crit_dmg_mult:.2f}x")

    print(f"\n[ REWARDS ]")
    print(f"Exp Gain Mult:       {player.exp_gain_mult:.2f}x")
    print(f"Frag/Loot Gain Mult: {player.frag_loot_gain_mult:.2f}x")

    print(f"\n[ MODIFIERS ]")
    print(f"Stamina Mod Ch.:   {player.stamina_mod_chance*100:.2f}%")
    print(f"Stamina Mod Gain:  {player.stamina_mod_gain:.2f}x")
    
    print_debug_equations(player)


if __name__ == "__main__":
    p = Player()
    
    # Store the JSON file right next to this verification script
    json_path = os.path.join(SCRIPT_DIR, "player_state.json")
    
    if not os.path.exists(json_path):
        generate_json_template(Player, json_path)
        print(f"\n[INIT] A blank JSON template has been created at:\n  {json_path}")
        print("Please fill it with your data and run the script again.")
    else:
        # Backward compatibility check
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'arch_ability_infernal_bonus' not in data['settings']:
            print("Detected an old JSON template format. Deleting and rebuilding...")
            os.remove(json_path)
            generate_json_template(Player, json_path)
            print("Please re-enter your data into the newly generated JSON file!")
        else:
            load_state_from_json(p, json_path)
            print_character_sheet(p)