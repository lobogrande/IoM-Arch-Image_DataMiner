# ==============================================================================
# Script: tools/verify_player.py
# Layer 3: State Management & Translation
# Description: Loads and Saves player_state.json. Features a robust hybrid-key 
#              parser ("3 - Gem Stamina") to make JSON files human-readable 
#              while avoiding duplicate-key JSON overwrites. Supports dynamic
#              pruning of locked progression stats.
# ==============================================================================

import json
import os
import sys

# --- BULLETPROOF PATHING ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Add the parent directory so it can find project_config.py
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.player import Player
import project_config as cfg 

def load_state_from_json(player: Player, filepath: str):
    """
    Loads JSON data into the Player object. 
    Supports legacy integer keys ("3"), hybrid keys ("3 - Gem Stamina"),
    and unified logical groups for external upgrades.
    """
    if not os.path.exists(filepath):
        print(f"[Warning] Save file not found at {filepath}. Generating baseline template...")
        save_state_to_json(player, filepath, hide_locked=True) 
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
        reverse_external = {val[1]: key for key, val in player.EXTERNAL_DEF.items()}
        
        for k, v in data['external_upgrades'].items():
            matched_group = next((g for g in cfg.EXTERNAL_UI_GROUPS if g["name"] == k), None)
            if matched_group:
                for r in matched_group["rows"]:
                    player.set_external_level(r, v)
                continue
                
            upgrade_id = parse_key(k)
            if upgrade_id is not None:
                player.set_external_level(upgrade_id, v)
            elif k in reverse_external:
                player.set_external_level(reverse_external[k], v)

    # 5. Load Cards
    if 'cards' in data:
        for card_id, lvl in data['cards'].items():
            player.set_card_level(card_id, lvl)
            
    return True

def save_state_to_json(player: Player, filepath: str, readable_keys: bool = True, hide_locked: bool = False):
    """
    Generates a player_state.json file. 
    If readable_keys is True, formats dictionaries as "ID - Name" for UX.
    If hide_locked is True, removes Asc2-only data if the player hasn't unlocked it.
    """
    
    # Prune Settings
    settings_out = {
        "asc2_unlocked": player.asc2_unlocked,
        "arch_level": player.arch_level,
        "current_max_floor": player.current_max_floor,
        "base_damage_const": player.base_damage_const,
        "hades_idol_level": player.hades_idol_level,
        "total_infernal_cards": player.total_infernal_cards,
        "arch_ability_infernal_bonus": player.arch_ability_infernal_bonus
    }
    if hide_locked and not player.asc2_unlocked:
        if 'hades_idol_level' in settings_out:
            del settings_out['hades_idol_level']

    # Prune Base Stats
    base_stats_out = player.base_stats.copy()
    if hide_locked and not player.asc2_unlocked:
        if 'Corr' in base_stats_out:
            del base_stats_out['Corr']
            
    # Prune Cards
    cards_out = {}
    for k, v in player.cards.items():
        if hide_locked and not player.asc2_unlocked and k.endswith('4'):
            continue
        cards_out[k] = v

    data = {
        "settings": settings_out,
        "base_stats": base_stats_out,
        "internal_upgrades": {},
        "external_upgrades": {},
        "cards": cards_out
    }

    asc2_locked_rows =[17, 19, 34, 46, 52, 55]

    # Populate Internal Upgrades
    for k, v in player.upgrade_levels.items():
        if hide_locked and not player.asc2_unlocked and k in asc2_locked_rows:
            continue
            
        if readable_keys and k in player.UPGRADE_DEF:
            name = player.UPGRADE_DEF[k][0]
            data["internal_upgrades"][f"{k} - {name}"] = v
        else:
            data["internal_upgrades"][str(k)] = v

    # Populate External Upgrades
    for group in cfg.EXTERNAL_UI_GROUPS:
        representative_val = player.external_levels.get(group["rows"][0], 0)
        data["external_upgrades"][group["name"]] = representative_val

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ State saved successfully to {filepath}")

if __name__ == "__main__":
    test_player = Player()
    json_path = os.path.join(BASE_DIR, "tools", "player_state.json")
    
    load_state_from_json(test_player, json_path)
    save_state_to_json(test_player, json_path, readable_keys=True, hide_locked=True)
    print(f"Verification Complete. Player Max Sta: {test_player.max_sta}")