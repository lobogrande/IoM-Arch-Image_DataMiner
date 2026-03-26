# ==============================================================================
# Script: core/ore.py
# Version: 1.0.1 (Modular Architecture)
# Description: Generates a distinct Ore object containing the final calculated 
#              HP, Armor, XP, and Loot yields based on the floor tier scaling 
#              and the player's card multipliers.
# ==============================================================================

import sys
import os
import math

# --- BULLETPROOF PATHING ---
# 1. Add the 07_Modeling_and_Simulation directory to path
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

# 2. Add the Project Root directory to path (for project_config.py)
ROOT_DIR = os.path.abspath(os.path.join(SIM_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import project_config as cfg

class Ore:
    def __init__(self, ore_id, current_floor, player):
        self.ore_id = ore_id
        self.current_floor = current_floor
        
        # Pull Base Stats from configuration
        base = cfg.ORE_BASE_STATS.get(ore_id, None)
        if not base:
            raise ValueError(f"Ore ID '{ore_id}' not found in project_config.py!")

        # Pull Card Multipliers from the Player engine
        hp_mult, exp_mult, loot_mult = player.get_card_bonuses(ore_id)

        # 1. Determine Floor Scaling for HP and Armor
        if player.asc2_unlocked and current_floor >= 150:
            raw_hp, self.armor = base['hp150'], base['a150']
        elif current_floor >= 100:
            raw_hp, self.armor = base['hp100'], base['a100']
        else:
            raw_hp, self.armor = base['hp'], base['a']

        # Apply Card HP Reduction (Standard game rounding is usually nearest integer)
        self.hp = round(raw_hp * hp_mult)

        # 2. Calculate XP Yield
        base_xp = base['xp']
        p_exp_mult = player.exp_gain_mult
        
        # Note: 'exp_mult' from cards already includes the (1 + X%) logic.
        raw_exp = base_xp * p_exp_mult * exp_mult
        
        # Excel: FLOOR(..., 0.001)
        floored_exp = math.floor((raw_exp + 1e-9) * 1000) / 1000.0
        
        # Excel: IF(exp>100, ROUNDDOWN(exp, 0), exp)
        if floored_exp > 100.0:
            self.xp = math.floor(floored_exp + 1e-9)
        else:
            self.xp = floored_exp

        # 3. Calculate Fragment Yield
        base_frag = base['fa']
        p_frag_mult = player.frag_loot_gain_mult
        self.frag_type = base['ft']
        
        raw_frag = base_frag * p_frag_mult * loot_mult
        
        # Excel: ROUND(..., 3)
        self.frag_amt = math.floor((raw_frag + 1e-9) * 1000 + 0.5) / 1000.0


# ==============================================================================
# QUICK VERIFICATION TEST
# ==============================================================================
if __name__ == "__main__":
    from player import Player
    
    p = Player()
    # Let's give the player a Level 3 (Polychrome) Card for rare1
    p.set_card_level('rare1', 3)
    p.asc2_unlocked = False

    # Test Floor 50 (Base stats)
    ore_f50 = Ore('rare1', 50, p)
    print(f"--- RARE1 (Poly Card) @ Floor 50 ---")
    print(f"HP: {ore_f50.hp} | Armor: {ore_f50.armor} | XP: {ore_f50.xp:.3f} | Loot: {ore_f50.frag_amt:.3f}")

    # Test Floor 101 (Scaled stats)
    ore_f101 = Ore('rare1', 101, p)
    print(f"\n--- RARE1 (Poly Card) @ Floor 101 ---")
    print(f"HP: {ore_f101.hp} | Armor: {ore_f101.armor} | XP: {ore_f101.xp:.3f} | Loot: {ore_f101.frag_amt:.3f}")