# ==============================================================================
# Script: tools/verify_player.py
# Layer 3: State Management & Translation
# Description: Loads and Saves player_state.json. Features a robust hybrid-key 
#              parser ("3 - Gem Stamina") to make JSON files human-readable 
#              while avoiding duplicate-key JSON overwrites.
# ==============================================================================

import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player

def load_state_from_json(player: Player, filepath: str):
    """
    Loads JSON data into the Player object. 
    Supports legacy integer keys ("3") and hybrid keys ("3 - Gem Stamina").
    """
    if not os.path.exists(filepath):
        print(f"[Warning] Save file not found at {filepath}. Generating baseline template...")
        save_state_to_json(player, filepath) # Generate the missing template!
        return False

    with open(filepath, 'r') as f:
        data = json.load(f)

    # 1. Load Settings
    if 'settings' in data:
        s = data['settings']
        player.asc2_unlocked = s.get('asc2_unlocked', False)
        player.arch_level = s.get('arch_level', 1)
        player.current_max_floor = s.get('current_max_floor', 100)
        player.base_damage_const = s.get('base_damage_const', 10)
        player.hades_idol_level = s.get('hades_idol_level', 0)
        player.total_infernal_cards = s.get('total_infernal_cards', 0)
        player.arch_ability_infernal_bonus = s.get('arch_ability_infernal_bonus', 0.0)

    # 2. Load Base Stats
    if 'base_stats' in data:
        for stat, val in data['base_stats'].items():
            player.base_stats[stat] = val

    # Helper function to extract integer ID from hybrid keys ("3 - Gem Stamina" -> 3)
    def parse_key(k):
        try:
            return int(str(k).split(" - ")[0])
        except ValueError:
            return None

    # 3. Load Internal Upgrades
    if 'internal_upgrades' in data:
        for k, v in data['internal_upgrades'].items():
            upgrade_id = parse_key(k)
            if upgrade_id is not None:
                player.set_upgrade_level(upgrade_id, v)

    # 4. Load External Upgrades
    if 'external_upgrades' in data:
        for k, v in data['external_upgrades'].items():
            upgrade_id = parse_key(k)
            if upgrade_id is not None:
                player.set_external_level(upgrade_id, v)

    # 5. Load Cards
    if 'cards' in data:
        for card_id, lvl in data['cards'].items():
            player.set_card_level(card_id, lvl)
            
    return True

def save_state_to_json(player: Player, filepath: str, readable_keys: bool = True):
    """
    Generates a player_state.json file. 
    If readable_keys is True, formats dictionaries as "ID - Name" for UX.
    """
    data = {
        "settings": {
            "asc2_unlocked": player.asc2_unlocked,
            "arch_level": player.arch_level,
            "current_max_floor": player.current_max_floor,
            "base_damage_const": player.base_damage_const,
            "hades_idol_level": player.hades_idol_level,
            "total_infernal_cards": player.total_infernal_cards,
            "arch_ability_infernal_bonus": player.arch_ability_infernal_bonus
        },
        "base_stats": player.base_stats,
        "internal_upgrades": {},
        "external_upgrades": {},
        "cards": player.cards
    }

    # Populate Internal Upgrades
    for k, v in player.upgrade_levels.items():
        if readable_keys and k in player.UPGRADE_DEF:
            name = player.UPGRADE_DEF[k][0]
            data["internal_upgrades"][f"{k} - {name}"] = v
        else:
            data["internal_upgrades"][str(k)] = v

    # Populate External Upgrades
    for k, v in player.external_levels.items():
        if readable_keys and k in player.EXTERNAL_DEF:
            name = f"{player.EXTERNAL_DEF[k][0]} ({player.EXTERNAL_DEF[k][1]})"
            data["external_upgrades"][f"{k} - {name}"] = v
        else:
            data["external_upgrades"][str(k)] = v

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ State saved successfully to {filepath}")

if __name__ == "__main__":
    test_player = Player()
    json_path = os.path.join(BASE_DIR, "tools", "player_state.json")
    
    # Test the loader (will trigger the generator if file is missing)
    load_state_from_json(test_player, json_path)
    
    # Test the generator manually to rewrite the file with human-readable keys
    save_state_to_json(test_player, json_path, readable_keys=True)
    
    print(f"Verification Complete. Player Max Sta: {test_player.max_sta}")