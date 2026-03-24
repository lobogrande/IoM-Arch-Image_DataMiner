# ==============================================================================
# Script: ore_simulator_mockup.py
# Version: 2.1.0
# Description: Simulates floor generation using Gaussian stats and executes a 
#              serpentine pathing run-through. Features a fully-mapped Player 
#              class capable of parsing external JSON states for verification.
# ==============================================================================

import sys
import os
import json
import math
import pandas as pd

# Dynamically link to root-level project_config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

SCRIPT_DIR = os.path.dirname(__file__)

class Player:
    # ==========================================================================
    # INTERNAL UPGRADE DEFINITIONS (Row mappings to Base Multipliers)
    # ==========================================================================
    UPGRADE_DEF = {
        3:  ("Gem Stamina", 2.0, 0.0005),
        4:  ("Gem Exp", 0.05, 0.0005),
        5:  ("Gem Loot", 0.02, 0.0005),
        9:  ("Flat Damage", 1.0, None),
        10: ("Armor Pen.", 1.0, None),
        11: ("Exp. Gain", 0.02, None),
        12: ("Stat Points", 1.0, None),
        13: ("Crit Chance/Damage", 0.0025, 0.01),
        14: ("Max Sta/Sta Mod Chance", 2.0, 0.0005),
        15: ("Flat Damage", 2.0, None),
        16: ("Loot Mod Gain", 0.3, None),
        17: ("Unlock Fairy/Armor Pen", 3.0, None), # Asc2 Locked
        18: ("Enrage&Crit Dmg/Enrage Cooldown", 0.02, -1.0),
        19: ("Gleaming Floor Chance", 0.001, None), # Asc2 Locked
        20: ("Flat Dmg/Super Crit Chance", 2.0, 0.0035),
        21: ("Exp Gain/Fragment Gain", 0.03, 0.02),
        22: ("Flurry Sta Gain/Flurr Cooldown", 1.0, -1.0),
        23: ("Max Sta/Sta Mod Gain", 4.0, 1.0),
        24: ("All Mod Chances", 0.0002, None),
        25: ("Flat Dmg/Damage Up", 0.2, 0.001),
        26: ("Max Sta/Mod Chance", 1.0, 0.0002),
        27: ("Unlock Ability Fairy/Loot Mod Gain", 0.05, None),
        28: ("Exp Gain/Max Sta", 0.05, 0.01),
        29: ("Armor Pen/Ability Cooldowns", 0.02, -1.0),
        30: ("Crit Dmg/Super Crit Dmg", 0.02, 0.02),
        31: ("Quake Atks/Cooldown", 1.0, -2.0),
        32: ("Flat Dmg/Enrage Cooldown", 3.0, -1.0),
        33: ("Mod Chance/Armor Pen", 0.0001, 1.0),
        34: ("Buff Divinity[Div Stats Up]", 0.2, None), # Asc2 Locked
        35: ("Exp Gain/Mod Ch.", 0.01, 0.0001),
        36: ("Damage Up/Armor Pen", 0.02, 3.0),
        37: ("Super Crit/Ultra Crit Chance", 0.0035, 0.01),
        38: ("Exp Mod Gain/Chance", 0.1, 0.001),
        39: ("Ability Insta Chance/Max Sta", 0.003, 4.0),
        40: ("Ultra Crit Dmg/Sta Mod Chance", 0.02, 0.0003),
        41: ("Poly Card Bonus", 0.15, None),
        42: ("Frag Gain Mult", None, None),
        43: ("Sta Mod Gain", 2.0, None),
        44: ("All Mod Chances", 0.015, None),
        45: ("Exp Gain/All Stat Cap Inc.", 2.0, 5.0),
        46: ("Gleaming Floor Multi", 0.03, None), # Asc2 Locked
        47: ("Damage Up/Crit Dmg Up", 0.01, 0.01),
        48: ("Gold Crosshair Chance/Auto-Tap Chance", 0.01, 0.01),
        49: ("Flat Dmg/Ultra Crit Chance", 3.0, 0.005),
        50: ("Ability Insta Chance/Sta Mod Chance", 0.001, 0.001),
        51: ("Dmg Up/Exp Gain", 0.1, 0.1),
        # FIXED: Row 52 Update to Mod Multi Up
        52: ("[Corruption Buff] Dmg Up / Mod Multi Up", 0.002, 0.0002), # Asc2 Locked
        53: ("Super Crit Dmg/Exp Mod Gain", 0.005, 0.02),
        54: ("Max Sta/Crosshair Auto-Tap Chance", 0.005, 0.002),
        55: ("All Mod Multipliers", 0.02, None) # Asc2 Locked
    }

    EXTERNAL_DEF = {
        4:  ("Idol", "Hestia (Fragment Gain)"),
        5:  ("Pet Skin", "Axolotl (Fragment Gain)"),
        6:  ("Pet Skin", "Dino (Astraeus&Chione Idol Cap)"),
        7:  ("Pet Skin", "Dino (Aphrodite&Tethys Idol Cap)"),
        8:  ("Leg Fish Tribute", "Geoduck (Frag Gain per Myth Chest Owned)"),
        9:  ("Skill Tree", "Avada Keda- Skill (Ability Duration)"),
        10: ("Skill Tree", "Avada Keda- Skill (Ability Cooldown)"),
        11: ("Skill Tree", "Avada Keda- Skill (Ability Instacharge Chance)"),
        12: ("Skill Tree", "Block Bonker (Dmg per Highest Floor)"),
        13: ("Skill Tree", "Block Bonker (Max Sta per Highest Floor)"),
        14: ("Skill Tree", "Block Bonker (Speed Mod Gain per Highest Floor)"),
        15: ("Store VPs", "Archaeology Bundle (Frag Gain)"),
        16: ("Store VPs", "Ascension Bundle VP (Exp Mult)"),
        17: ("Store VPs", "Ascension Bundle VP (Crosshair Auto-Tap Chance)"),
        18: ("Store VPs", "Ascension Bundle VP (Loot Mod Chance)"),
        19: ("Store VPs", "Ascension Bundle VP (Golden Crosshair Chance)"),
        20: ("Cards", "Arch Ability Misc Card (Ability Cooldown Reduction)")
    }

    def __init__(self):
        self.asc2_unlocked = False     
        self.arch_level = 1            
        self.current_max_floor = 100   
        self.base_damage_const = 10    
        self.infernal_card_bonus = 0.0 
        
        self.hades_idol_level = 0
        self.total_infernal_cards = 0
        
        self.base_stats = {
            'Str': 0, 'Agi': 0, 'Per': 0, 'Int': 0, 'Luck': 0, 'Div': 0, 'Corr': 0
        }

        self.upgrade_levels = {}
        self.upgrades = {}
        self.external_levels = {}
        self.external = {}
        
        for row in self.UPGRADE_DEF.keys():
            self.set_upgrade_level(row, 0)
            
        for row in self.EXTERNAL_DEF.keys():
            self.set_external_level(row, 0)
            
        self._init_cards()

    # --------------------------------------------------------------------------
    # ENGINE VALUE SETTERS
    # --------------------------------------------------------------------------
    def set_upgrade_level(self, row, lvl):
        self.upgrade_levels[row] = lvl
        if row == 42:
            self.upgrades['F42'] = 1.0 if lvl == 0 else 1.25
            return

        if row in self.UPGRADE_DEF:
            name, f_mult, h_mult = self.UPGRADE_DEF[row]
            if f_mult is not None: self.upgrades[f'F{row}'] = lvl * f_mult
            if h_mult is not None: self.upgrades[f'H{row}'] = lvl * h_mult

    def set_external_level(self, row, lvl):
        self.external_levels[row] = lvl
        w = self.external
        if row == 4:  w['W4'] = lvl * 0.0001
        elif row == 5:  w['W5'] = (1.0 + lvl) * 0.03
        elif row == 6:  w['W6'] = (1.0 + lvl) * 50.0
        elif row == 7:  w['W7'] = (1.0 + lvl) * 30.0
        elif row == 8:  w['W8_raw'] = lvl * 0.0025
        elif row == 9:  w['W9'] = lvl * 5.0
        elif row == 10: w['W10'] = lvl * -10.0
        elif row == 11: w['W11'] = lvl * 0.03
        elif row == 12: w['W12'] = lvl * 0.01
        elif row == 13: w['W13'] = lvl * 0.01
        elif row == 14: w['W14'] = lvl * 1.0
        elif row == 15: w['W15'] = max(1.0, lvl * 1.25)
        elif row == 16: w['W16'] = max(1.0, lvl * 1.15)
        elif row == 17: w['W17'] = lvl * 0.05
        elif row == 18: w['W18'] = lvl * 0.02
        elif row == 19: w['W19'] = lvl * 0.02
        elif row == 20: 
            if lvl == 0:   w['W20'] = 0.0
            elif lvl == 1: w['W20'] = -0.03
            elif lvl == 2: w['W20'] = -0.06
            elif lvl == 3: w['W20'] = -0.10
            elif lvl == 4: w['W20'] = self.infernal_card_bonus

    # --------------------------------------------------------------------------
    # DYNAMIC VALUE GETTERS & TOGGLES
    # --------------------------------------------------------------------------
    def u(self, cell): 
        if not self.asc2_unlocked:
            locked_rows =[17, 19, 34, 46, 52, 55]
            try:
                if int(cell[1:]) in locked_rows:
                    return 0.0
            except ValueError:
                pass
        return self.upgrades.get(cell, 0.0)

    def w(self, cell, default=0.0): 
        if cell == 'W8':
            cap = 0.75 if self.asc2_unlocked else 0.50
            return min(cap, self.external.get('W8_raw', 0.0))
        return self.external.get(cell, default)
    
    def stat(self, stat_name):
        if not self.asc2_unlocked and stat_name == 'Corr':
            return 0.0
        return self.base_stats.get(stat_name, 0.0)

    # ==========================================================================
    # CARDS SYSTEM & INFERNAL ENGINE
    # ==========================================================================
    def _init_cards(self):
        self.cards = {}
        for ot in['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']:
            for tier in range(1, 5):
                self.cards[f"{ot}{tier}"] = 0

    def set_card_level(self, ore_id, lvl):
        self.cards[ore_id] = lvl

    def get_card_bonuses(self, ore_id):
        if ore_id.endswith('4') and not self.asc2_unlocked:
            return 1.0, 1.0, 1.0

        lvl = self.cards.get(ore_id, 0)
        hp_mult, exp_mult, loot_mult = 1.0, 1.0, 1.0
        
        if lvl == 1:
            hp_mult, exp_mult, loot_mult = 0.90, 1.10, 1.10
        elif lvl == 2:
            hp_mult, exp_mult, loot_mult = 0.80, 1.20, 1.20
        elif lvl >= 3:
            poly_bonus = 0.35 + self.u('F41') 
            hp_mult = 1.0 - poly_bonus
            exp_mult = 1.0 + poly_bonus
            loot_mult = 1.0 + poly_bonus

        return hp_mult, exp_mult, loot_mult

    @property
    def arch_infernal_cards(self):
        return sum(1 for lvl in self.cards.values() if lvl == 4)

    @property
    def infernal_multiplier(self):
        hades_bonus = (self.hades_idol_level * 0.000045) if self.asc2_unlocked else 0.0
        arch_bonus = 1.0 + (0.04 * self.arch_infernal_cards) + (0.002 * self.total_infernal_cards)
        return math.ceil(arch_bonus * (1.0 + hades_bonus) * 10000) / 10000.0

    def inf(self, ore_id):
        if ore_id.endswith('4') and not self.asc2_unlocked:
            return 0.0
            
        if self.cards.get(ore_id, 0) == 4:
            inf_mult = self.infernal_multiplier
            bases = {
                'dirt1': (0.1, 4), 'dirt2': (0.12, 4), 'dirt3': (0.08, 4),
                'com1': (0.06, 4), 'com2': (0.07, 4), 'com3': (0.08, 4),
                'rare1': (0.05, 4), 'rare2': (20.0, 0), 'rare3': (0.4, 4),
                'epic1': (0.3, 4), 'epic2': (0.04, 4), 'epic3': (0.05, 4),
                'leg1': (0.04, 4), 'leg2': (0.05, 4), 'leg3': (40.0, 0),
                'myth1': (0.013, 4), 'myth2': (0.008, 4), 'myth3': (0.007, 4),
                'div1': (0.1, 4), 'div2': (0.0125, 4), 'div3': (1.126, 0)
            }
            if ore_id in bases:
                base_val, decimals = bases[ore_id]
                return round(base_val * inf_mult, decimals)
        return 0.0

    # ==========================================================================
    # 1. COMBAT CALCULATIONS
    # ==========================================================================
    @property
    def max_sta(self):
        base_calc = 100 + self.u('F14') + self.u('F23') + self.u('H39') + self.u('F3')
        stat_calc = self.stat('Agi') * (5 + self.u('F26'))
        asc2_calc = (1 + self.u('H28') + self.u('F54')) * (1 - 0.03 * self.stat('Corr'))
        floor_calc = 1 + (0.01 * min(100, self.current_max_floor))
        val = (base_calc + stat_calc) * asc2_calc * floor_calc * (1.0 + self.inf('epic3'))
        return round(val)

    @property
    def damage(self):
        base_calc = self.u('F9') + self.u('F15') + self.u('F20') + self.u('F32') + self.u('F49') + self.inf('rare2')
        stat_calc1 = self.stat('Str') * (1 + self.u('F25'))
        stat_calc2 = self.stat('Div') * (2 + self.u('F34'))
        mult1 = 1 + self.u('F51') + self.u('F36') + (self.stat('Str') * (0.01 + self.u('F47') + self.u('H25'))) + self.inf('div1')
        mult2 = (0.06 + self.u('F52')) * self.stat('Corr')
        floor_calc = 1 + (0.01 * min(100, self.current_max_floor))
        val = (base_calc + stat_calc1 + stat_calc2 + self.base_damage_const) * (mult1 + mult2) * floor_calc
        return round(val)

    @property
    def armor_pen(self):
        stat_calc = self.stat('Per') * (2 + self.u('H33'))
        base_ap = self.u('F10') + self.u('F17') + self.u('H36') + stat_calc + self.inf('leg3')
        mult_ap = 1 + (0.03 * self.stat('Int')) + self.u('F29') + self.inf('rare3')
        return round(base_ap * mult_ap)

    @property
    def atk_spd(self): return 1.0

    # ==========================================================================
    # 2. CRITICAL HIT SYSTEM
    # ==========================================================================
    @property
    def crit_chance(self): return self.u('F13') + (0.02 * self.stat('Luck')) + (0.01 * self.stat('Agi'))
    @property
    def crit_dmg_mult(self):
        inner_calc = 1.0 + self.u('H13') + self.u('F30') + self.inf('com1')
        stat_calc = (0.03 + self.u('H47')) * self.stat('Str')
        return math.floor((1.5 * (inner_calc + stat_calc)) * 100) / 100.0
    @property
    def super_crit_chance(self): return self.u('H20') + self.u('F37') + ((0.02 + 0.01 * self.u('F34')) * self.stat('Div')) + self.inf('epic2')
    @property
    def super_crit_dmg_mult(self):
        if self.super_crit_chance > 0: return 2.0 * (1.0 + self.u('H30') + self.u('F53') + self.inf('com2'))
        return 0.0
    @property
    def ultra_crit_chance(self): return self.u('H37') + self.u('H49')
    @property
    def ultra_crit_dmg_mult(self):
        if self.ultra_crit_chance > 0: return 3.0 * (1.0 + self.u('F40')) * (1.0 + self.inf('com3'))
        return 0.0

    # ==========================================================================
    # 3. ABILITY & CROSSHAIR MECHANICS
    # ==========================================================================
    @property
    def ability_insta_charge(self): return self.w('W11') + self.u('F39') + self.u('F50')
    @property
    def crosshair_auto_tap(self): return 0.05 + self.u('H48') + self.u('H54') + ((0.02 + 0.01 * self.u('F34')) * self.stat('Div')) + self.inf('rare1')
    @property
    def gold_crosshair_chance(self): return 0.02 + self.u('F48') + (0.005 * self.stat('Luck')) + self.inf('leg2')
    @property
    def gold_crosshair_mult(self): return 3.0 + self.inf('epic1')

    # ==========================================================================
    # 4. REWARD MULTIPLIERS
    # ==========================================================================
    @property
    def exp_gain_mult(self):
        stat_calc = self.stat('Int') * (0.05 + self.u('F35'))
        val = (1 + self.u('F4') + self.u('F11') + self.u('F21') + self.u('F28') + self.u('H51') + stat_calc)
        val *= (1 + self.u('F45')) * self.w('W16', default=1.0) * (1.0 + self.inf('dirt2'))
        return math.floor(val * 100) / 100.0
    @property
    def frag_loot_gain_mult(self):
        stat_calc = self.stat('Per') * 0.04
        val = (1 + self.u('F5') + self.u('H21') + stat_calc)
        val *= (1 + self.w('W4')) * (1 + self.w('W5')) * (1 + min(0.75, self.w('W8')))
        val *= self.u('F42') * self.w('W15', default=1.0) * (1.0 + self.inf('dirt3') + self.inf('leg1'))
        return round(val, 2)

    # ==========================================================================
    # 5. MODIFIERS
    # ==========================================================================
    @property
    def exp_mod_chance(self): return self.u('H38') + self.u('H4') + (0.002 * self.stat('Luck')) + (0.0035 * self.stat('Int')) + self.u('F24') + self.u('F44')
    @property
    def exp_mod_gain(self): return (3.0 + self.u('F38') + self.u('H53')) * (1.0 + self.u('F55') + self.stat('Corr') * (0.01 + self.u('H52')))
    @property
    def loot_mod_chance(self): return self.u('H5') + self.u('F24') + self.u('F44') + self.w('W18') + (0.0035 * self.stat('Per')) + (0.002 * self.stat('Luck')) + self.inf('myth2')
    @property
    def loot_mod_gain(self): return (2.0 + self.u('F16') + self.u('F27')) * (1.0 + self.u('F55') + self.stat('Corr') * (0.01 + self.u('H52'))) * (1.0 + self.inf('dirt1'))
    @property
    def speed_mod_chance(self): return self.u('F24') + self.u('F44') + (0.003 * self.stat('Agi')) + (0.002 * self.stat('Luck'))
    @property
    def speed_mod_gain(self): return round(25.0 * (1.0 + self.u('F55') + self.stat('Corr') * (0.01 + self.u('H52'))))
    @property
    def speed_mod_attack_rate(self): return 2.0
    @property
    def stamina_mod_chance(self): return self.u('H3') + self.u('H14') + self.u('F24') + self.u('F44') + self.u('H40') + self.u('H50') + (0.002 * self.stat('Luck')) + self.inf('myth3')
    @property
    def stamina_mod_gain(self): return round((3.0 + self.u('F43') + self.u('H23')) * (1.0 + self.u('F55') + self.stat('Corr') * (0.01 + self.u('H52')))) + self.inf('div3')

    # ==========================================================================
    # 6. GLEAMING FLOOR MECHANIC
    # ==========================================================================
    @property
    def gleaming_floor_chance(self): return (self.u('F19') + self.inf('myth1') + self.inf('div2')) if self.asc2_unlocked else 0.0
    @property
    def gleaming_floor_multi(self): return (3.0 + self.u('F46')) if self.asc2_unlocked else 1.0

    # ==========================================================================
    # 7. SKILLS (Enrage, Flurry, Quake)
    # ==========================================================================
    @property
    def enrage_charges(self): return 5 + self.w('W9')
    @property
    def enrage_cooldown(self): return math.floor((60 + self.u('H18') + self.u('H29') + self.u('H32') + self.w('W10')) * (1 + self.w('W20')) * 10) / 10.0
    @property
    def enrage_bonus_dmg(self): return 0.2 + self.u('F18')
    @property
    def enrage_bonus_crit_dmg(self): return 1.0 + self.u('F18')
    @property
    def flurry_duration(self): return 5 + self.w('W9')
    @property
    def flurry_cooldown(self): return (120 + self.u('H22') + self.u('H29') + self.w('W10')) * (1 + self.w('W20'))
    @property
    def flurry_bonus_atk_spd(self): return 1.0
    @property
    def flurry_sta_on_cast(self): return 5 + self.u('F22')
    @property
    def quake_attacks(self): return 5 + self.u('F31') + self.w('W9')
    @property
    def quake_cooldown(self): return (180 + self.u('H29') + self.u('H31') + self.w('W10')) * (1 + self.w('W20'))
    @property
    def quake_dmg_to_all(self): return 0.2

    # ==========================================================================
    # DEBUG & VERIFICATION METHODS
    # ==========================================================================
    def generate_json_template(self, filepath):
        """Generates a blank JSON file for the user to fill out."""
        template = {
            "settings": {
                "asc2_unlocked": False,
                "arch_level": 1,
                "current_max_floor": 100,
                "base_damage_const": 10,
                "hades_idol_level": 0,
                "total_infernal_cards": 0,
                "infernal_card_bonus": 0.0
            },
            "base_stats": {
                "Str": 0, "Agi": 0, "Per": 0, "Int": 0, "Luck": 0, "Div": 0, "Corr": 0
            },
            "internal_upgrades": {str(k): 0 for k in self.UPGRADE_DEF.keys()},
            "external_upgrades": {str(k): 0 for k in self.EXTERNAL_DEF.keys()},
            "cards": {k: 0 for k in self.cards.keys()}
        }
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=4)

    def load_state_from_json(self, filepath):
        """Loads a user's JSON file and populates the Player engine."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # 1. Load Settings
        s = data.get('settings', {})
        self.asc2_unlocked = s.get('asc2_unlocked', False)
        self.arch_level = s.get('arch_level', 1)
        self.current_max_floor = s.get('current_max_floor', 100)
        self.base_damage_const = s.get('base_damage_const', 10)
        self.hades_idol_level = s.get('hades_idol_level', 0)
        self.total_infernal_cards = s.get('total_infernal_cards', 0)
        self.infernal_card_bonus = s.get('infernal_card_bonus', 0.0)
        
        # 2. Load Base Stats
        for stat, val in data.get('base_stats', {}).items():
            self.base_stats[stat] = val
            
        # 3. Load Internal Upgrades
        for row, lvl in data.get('internal_upgrades', {}).items():
            self.set_upgrade_level(int(row), lvl)
            
        # 4. Load External Upgrades
        for row, lvl in data.get('external_upgrades', {}).items():
            self.set_external_level(int(row), lvl)
            
        # 5. Load Cards
        for ore_id, lvl in data.get('cards', {}).items():
            self.set_card_level(ore_id, lvl)

    def print_character_sheet(self):
        """Prints a highly formatted layout of all calculated stats to terminal."""
        print("\n" + "="*50)
        print(" 🎮 PLAYER STATE VERIFICATION REPORT 🎮")
        print("="*50)
        
        print(f"\n[ COMBAT STATS ]")
        print(f"Max Stamina:       {self.max_sta:,}")
        print(f"Damage:            {self.damage:,}")
        print(f"Armor Pen:         {self.armor_pen:,}")
        print(f"Attack Speed:      {self.atk_spd:.2f}")

        print(f"\n[ CRITICAL SYSTEM ]")
        print(f"Crit Chance:       {self.crit_chance*100:.2f}%")
        print(f"Crit Dmg Mult:     {self.crit_dmg_mult:.2f}x")
        print(f"S. Crit Chance:    {self.super_crit_chance*100:.2f}%")
        print(f"S. Crit Dmg Mult:  {self.super_crit_dmg_mult:.2f}x")
        print(f"U. Crit Chance:    {self.ultra_crit_chance*100:.2f}%")
        print(f"U. Crit Dmg Mult:  {self.ultra_crit_dmg_mult:.2f}x")

        print(f"\n[ ABILITIES & CROSSHAIR ]")
        print(f"Ability Instacharge: {self.ability_insta_charge*100:.2f}%")
        print(f"Crosshair Auto-Tap:  {self.crosshair_auto_tap*100:.2f}%")
        print(f"Gold Crosshair Ch.:  {self.gold_crosshair_chance*100:.2f}%")
        print(f"Gold Crosshair Mult: {self.gold_crosshair_mult:.2f}x")

        print(f"\n[ REWARDS ]")
        print(f"Exp Gain Mult:       {self.exp_gain_mult:.2f}x")
        print(f"Frag/Loot Gain Mult: {self.frag_loot_gain_mult:.2f}x")

        print(f"\n[ MODIFIERS ]")
        print(f"Exp Mod Chance:    {self.exp_mod_chance*100:.2f}%")
        print(f"Exp Mod Gain:      {self.exp_mod_gain:.2f}x")
        print(f"Loot Mod Chance:   {self.loot_mod_chance*100:.2f}%")
        print(f"Loot Mod Gain:     {self.loot_mod_gain:.2f}x")
        print(f"Speed Mod Chance:  {self.speed_mod_chance*100:.2f}%")
        print(f"Speed Mod Gain:    {self.speed_mod_gain:.2f}x")
        print(f"Speed Mod AtkRate: {self.speed_mod_attack_rate:.2f}")
        print(f"Stamina Mod Ch.:   {self.stamina_mod_chance*100:.2f}%")
        print(f"Stamina Mod Gain:  {self.stamina_mod_gain:.2f}x")

        print(f"\n[ GLEAMING FLOOR ]")
        print(f"Gleam Chance:      {self.gleaming_floor_chance*100:.2f}%")
        print(f"Gleam Multiplier:  {self.gleaming_floor_multi:.2f}x")
        
        print(f"\n[ SKILLS ]")
        print(f"Enrage: {self.enrage_charges} Charges | {self.enrage_cooldown}s CD | +{self.enrage_bonus_dmg*100:.0f}% Dmg | +{self.enrage_bonus_crit_dmg*100:.0f}% CritDmg")
        print(f"Flurry: {self.flurry_duration}s Dur | {self.flurry_cooldown}s CD | +{self.flurry_bonus_atk_spd*100:.0f}% AtkSpd | +{self.flurry_sta_on_cast} Sta/Cast")
        print(f"Quake:  {self.quake_attacks} Attacks | {self.quake_cooldown}s CD | {self.quake_dmg_to_all*100:.0f}% AoE Dmg")
        print("="*50 + "\n")

# ------------------------------------------------------------------------------
# VERIFICATION RUNNER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    p = Player()
    
    # Define the expected path for the verification template
    json_path = os.path.join(SCRIPT_DIR, "player_state_template.json")
    
    if not os.path.exists(json_path):
        p.generate_json_template(json_path)
        print(f"\n[INIT] A blank JSON template has been created at:\n  {json_path}")
        print("\nPlease open the file, fill in your exact spreadsheet numbers, and run this script again to verify the math!")
    else:
        print(f"\n[INIT] Found {json_path}. Loading state...")
        p.load_state_from_json(json_path)
        p.print_character_sheet()