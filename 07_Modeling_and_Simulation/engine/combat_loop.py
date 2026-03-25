# ==============================================================================
# Script: engine/combat_loop.py
# Version: 1.0.1 (Modular Architecture)
# Description: The core simulation engine. Executes a run floor-by-floor using 
#              micro-tick hit-by-hit combat to perfectly simulate skill timers, 
#              speed attack pools, Quake splash damage, and exact crit rolls.
# ==============================================================================

import os
import sys
import random
import math

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from core.player import Player
from core.skills import SkillManager
from engine.floor_map import FloorGenerator

# --- COMBAT CONFIGURATION ---
STAMINA_COST_PER_ORE = 0.0   # No cost to approach an ore
STAMINA_COST_PER_HIT = 1.0   # 1 Stamina drained per pickaxe swing

# The exact 24-slot indices traversed in Serpentine Order
PATH_ORDER =[
    0, 1, 2, 3, 4, 5, 
    11, 10, 9, 8, 7, 6, 
    12, 13, 14, 15, 16, 17, 
    23, 22, 21, 20, 19, 18
]

class RunState:
    """Tracks the live 'bank accounts' and lifetime stats of a single simulation run."""
    def __init__(self, player):
        self.stamina = player.max_sta
        self.speed_pool = 0
        
        # Lifetime Analytics
        self.total_time = 0.0
        self.total_xp = 0.0
        self.total_frags = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        self.ores_mined = 0
        self.highest_floor = 1

class CombatSimulator:
    def __init__(self, player: Player):
        self.player = player
        self.generator = FloorGenerator()
        
    def _roll_crit_multiplier(self, skill_manager):
        """Rolls sequentially for the highest tier of crit, returning the multiplier."""
        # Roll Ultra
        if random.random() < self.player.ultra_crit_chance:
            return self.player.ultra_crit_dmg_mult
            
        # Roll Super
        if random.random() < self.player.super_crit_chance:
            return self.player.super_crit_dmg_mult
            
        # Roll Standard
        if random.random() < self.player.crit_chance:
            base_crit = self.player.crit_dmg_mult
            # Add Enrage Crit Bonus if active
            if skill_manager.is_enrage_active:
                base_crit += self.player.enrage_bonus_crit_dmg
            return base_crit
            
        return 1.0

    def _process_kill_rewards(self, ore, floor_obj, state: RunState):
        """Calculates and deposits XP, Loot, Stamina, and Speed Buffs upon ore death."""
        # 1. XP
        xp_yield = ore.xp * ore.modifiers.get('exp_multi', 1.0) * floor_obj.gleaming_multi
        state.total_xp += xp_yield
        
        # 2. Loot (Fragments)
        loot_yield = ore.frag_amt * ore.modifiers.get('loot_multi', 1.0) * floor_obj.gleaming_multi
        if ore.frag_type in state.total_frags:
            state.total_frags[ore.frag_type] += loot_yield
            
        # 3. Stamina Recovery
        sta_gain = ore.modifiers.get('stamina_gain', 0.0)
        if sta_gain > 0:
            state.stamina = min(self.player.max_sta, state.stamina + sta_gain)
            
        # 4. Speed Pool Top-up
        if ore.modifiers.get('speed_active', False):
            state.speed_pool += ore.modifiers.get('speed_gain', 0.0)
            
        state.ores_mined += 1

    def run_simulation(self):
        """Executes the combat loop across floors until stamina is exhausted."""
        state = RunState(self.player)
        skills = SkillManager(self.player)
        current_floor_id = 1
        
        print("\n[ SIMULATION STARTED ]")
        
        while state.stamina > 0:
            # Generate the environment
            floor = self.generator.generate_floor(current_floor_id, self.player)
            state.highest_floor = current_floor_id
            
            # Walk the Serpentine Path
            for i, slot_idx in enumerate(PATH_ORDER):
                if state.stamina <= 0:
                    break
                    
                target_ore = floor.grid[slot_idx]
                if target_ore is None or target_ore.hp <= 0:
                    continue # Empty slot or killed early by Quake splash
                    
                # Pay the entry cost to approach this ore (Now 0.0)
                state.stamina -= STAMINA_COST_PER_ORE
                
                # --- HIT-BY-HIT MICRO-TICK LOOP ---
                while target_ore.hp > 0 and state.stamina > 0:
                    # 1. Determine Attack Speed for this single hit
                    flurry_mult = 1.0 + self.player.flurry_bonus_atk_spd if skills.is_flurry_active else 1.0
                    
                    if state.speed_pool > 0:
                        current_atk_spd = self.player.atk_spd * target_ore.modifiers.get('speed_atk_rate', 1.0) * flurry_mult
                        state.speed_pool -= 1
                    else:
                        current_atk_spd = self.player.atk_spd * flurry_mult
                        
                    # Calculate how much time this 1 swing took
                    time_passed = 1.0 / current_atk_spd
                    state.total_time += time_passed
                    
                    # 2. Advance time and skill timers
                    events = skills.tick(time_passed)
                    
                    # Flurry restores stamina instantly upon auto-cast
                    if events["stamina_restored"] > 0:
                        state.stamina = min(self.player.max_sta, state.stamina + events["stamina_restored"])
                        
                    # 3. Calculate Damage for this hit
                    crit_mult = self._roll_crit_multiplier(skills)
                    
                    base_dmg = self.player.damage
                    if skills.is_enrage_active:
                        base_dmg *= (1.0 + self.player.enrage_bonus_dmg)
                        
                    actual_dmg = max(1.0, base_dmg - target_ore.armor) * crit_mult
                    target_ore.hp -= actual_dmg
                    
                    # Pay hit stamina cost (1.0 per swing)
                    state.stamina -= STAMINA_COST_PER_HIT
                    
                    # 4. Trigger Skills (Quake Splash)
                    quake_triggered = skills.consume_attack()
                    if quake_triggered:
                        # Find all alive ores in the remaining path
                        for bg_idx in PATH_ORDER[i+1:]:
                            bg_ore = floor.grid[bg_idx]
                            if bg_ore is not None and bg_ore.hp > 0:
                                q_crit = self._roll_crit_multiplier(skills)
                                # Quake does 20% of base damage minus armor, scaled by crit
                                q_dmg = max(1.0, (self.player.damage * self.player.quake_dmg_to_all) - bg_ore.armor) * q_crit
                                bg_ore.hp -= q_dmg
                                
                                # If Quake kills an ore in the background, harvest it instantly!
                                if bg_ore.hp <= 0:
                                    self._process_kill_rewards(bg_ore, floor, state)
                                    
                # Target Ore is dead. Harvest it!
                if target_ore.hp <= 0:
                    self._process_kill_rewards(target_ore, floor, state)
                    
            # Floor Complete!
            current_floor_id += 1
            
        print(f"[ SIMULATION FINISHED ]")
        print(f"Reached Floor: {state.highest_floor}")
        print(f"Ores Mined:    {state.ores_mined:,}")
        print(f"Total XP:      {state.total_xp:,.2f}")
        print(f"Time Taken:    {state.total_time/60:.2f} Minutes")
        
        return state

# ==============================================================================
# QUICK VERIFICATION TEST
# ==============================================================================
if __name__ == "__main__":
    from tools.verify_player import load_state_from_json
    
    p = Player()
    
    # Try to load the user's spreadsheet stats
    json_path = os.path.join(BASE_DIR, "tools", "player_state.json")
    if os.path.exists(json_path):
        load_state_from_json(p, json_path)
        print(f"Loaded Player JSON State successfully. Starting Max Stamina: {p.max_sta}")
    else:
        print(f"Warning: {json_path} not found. Running with baseline Level 0 stats.")
        
    sim = CombatSimulator(p)
    result_state = sim.run_simulation()