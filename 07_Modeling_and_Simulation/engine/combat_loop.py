# ==============================================================================
# Script: engine/combat_loop.py
# Version: 1.1.0 (Modular Architecture)
# Description: The core simulation engine. Executes a run floor-by-floor using 
#              micro-tick hit-by-hit combat. Now features embedded telemetry 
#              tracking for diagnostic visualization and hit-type analysis.
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

STAMINA_COST_PER_ORE = 0.0
STAMINA_COST_PER_HIT = 1.0

PATH_ORDER =[
    0, 1, 2, 3, 4, 5, 
    11, 10, 9, 8, 7, 6, 
    12, 13, 14, 15, 16, 17, 
    23, 22, 21, 20, 19, 18
]

class RunState:
    def __init__(self, player):
        self.stamina = player.max_sta
        self.speed_pool = 0
        self.total_time = 0.0
        self.total_xp = 0.0
        self.total_frags = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        self.ores_mined = 0
        self.specific_ores_mined = {} # <-- ADD THIS LINE
        self.highest_floor = 1
        
        # --- TELEMETRY DATA ---
        self.hit_counts = {'normal': 0, 'crit': 0, 'super': 0, 'ultra': 0}
        self.history = {
            'floor': [],
            'time': [],
            'stamina':[],
            'speed_pool':[]
        }

    def record_telemetry(self):
        """Snapshots the current state into the history arrays."""
        self.history['floor'].append(self.highest_floor)
        self.history['time'].append(self.total_time)
        self.history['stamina'].append(self.stamina)
        self.history['speed_pool'].append(self.speed_pool)

class CombatSimulator:
    def __init__(self, player: Player):
        self.player = player
        self.generator = FloorGenerator()
        
    def _roll_crit_multiplier(self, skill_manager):
        """Rolls sequentially for highest crit tier. Returns (multiplier, hit_type_string)."""
        if random.random() < self.player.ultra_crit_chance: 
            return self.player.ultra_crit_dmg_mult, 'ultra'
            
        if random.random() < self.player.super_crit_chance: 
            return self.player.super_crit_dmg_mult, 'super'
            
        if random.random() < self.player.crit_chance:
            base_crit = self.player.crit_dmg_mult
            if skill_manager.is_enrage_active: 
                base_crit += self.player.enrage_bonus_crit_dmg
            return base_crit, 'crit'
            
        return 1.0, 'normal'

    def _process_kill_rewards(self, ore, floor_obj, state: RunState):
        xp_yield = ore.xp * ore.modifiers.get('exp_multi', 1.0) * floor_obj.gleaming_multi
        state.total_xp += xp_yield
        
        loot_yield = ore.frag_amt * ore.modifiers.get('loot_multi', 1.0) * floor_obj.gleaming_multi
        if ore.frag_type in state.total_frags:
            state.total_frags[ore.frag_type] += loot_yield
            
        sta_gain = ore.modifiers.get('stamina_gain', 0.0)
        if sta_gain > 0:
            state.stamina = min(self.player.max_sta, state.stamina + sta_gain)
            
        if ore.modifiers.get('speed_active', False):
            state.speed_pool += ore.modifiers.get('speed_gain', 0.0)
            
        state.ores_mined += 1
        ore_id = ore.ore_id
        state.specific_ores_mined[ore_id] = state.specific_ores_mined.get(ore_id, 0) + 1
        
    def run_simulation(self):
        state = RunState(self.player)
        skills = SkillManager(self.player)
        current_floor_id = 1
        
        print("\n[ SIMULATION STARTED ]")
        state.record_telemetry() # Record starting state
        
        while state.stamina > 0:
            floor = self.generator.generate_floor(current_floor_id, self.player)
            state.highest_floor = current_floor_id
            
            for i, slot_idx in enumerate(PATH_ORDER):
                if state.stamina <= 0: break
                    
                target_ore = floor.grid[slot_idx]
                if target_ore is None or target_ore.hp <= 0: continue
                    
                state.stamina -= STAMINA_COST_PER_ORE
                
                while target_ore.hp > 0 and state.stamina > 0:
                    flurry_mult = 1.0 + self.player.flurry_bonus_atk_spd if skills.is_flurry_active else 1.0
                    
                    if state.speed_pool > 0:
                        current_atk_spd = self.player.atk_spd * self.player.speed_mod_attack_rate * flurry_mult
                        state.speed_pool -= 1
                    else:
                        current_atk_spd = self.player.atk_spd * flurry_mult
                        
                    time_passed = 1.0 / current_atk_spd
                    state.total_time += time_passed
                    
                    events = skills.tick(time_passed)
                    if events["stamina_restored"] > 0:
                        state.stamina = min(self.player.max_sta, state.stamina + events["stamina_restored"])
                        
                    crit_mult, crit_type = self._roll_crit_multiplier(skills)
                    state.hit_counts[crit_type] += 1  # Record telemetry
                    
                    base_dmg = self.player.damage
                    if skills.is_enrage_active: base_dmg *= (1.0 + self.player.enrage_bonus_dmg)
                        
                    actual_dmg = max(1.0, base_dmg - target_ore.armor) * crit_mult
                    target_ore.hp -= actual_dmg
                    state.stamina -= STAMINA_COST_PER_HIT
                    
                    if skills.consume_attack():
                        for bg_idx in PATH_ORDER[i+1:]:
                            bg_ore = floor.grid[bg_idx]
                            if bg_ore is not None and bg_ore.hp > 0:
                                q_crit, q_type = self._roll_crit_multiplier(skills)
                                state.hit_counts[q_type] += 1 # Record splash telemetry
                                
                                q_dmg = max(1.0, (self.player.damage * self.player.quake_dmg_to_all) - bg_ore.armor) * q_crit
                                bg_ore.hp -= q_dmg
                                if bg_ore.hp <= 0:
                                    self._process_kill_rewards(bg_ore, floor, state)
                                    
                if target_ore.hp <= 0:
                    self._process_kill_rewards(target_ore, floor, state)
                
                # Record state after every slot processed for high-res graphing
                state.record_telemetry()
                    
            current_floor_id += 1
            
        print(f"[ SIMULATION FINISHED ]")
        print(f"Reached Floor: {state.highest_floor}")
        print(f"Ores Mined:    {state.ores_mined:,}")
        print(f"Total XP:      {state.total_xp:,.2f}")
        print(f"Time Taken:    {state.total_time/60:.2f} Minutes")
        
        return state

if __name__ == "__main__":
    from tools.verify_player import load_state_from_json
    
    p = Player()
    json_path = os.path.join(BASE_DIR, "tools", "player_state.json")
    if os.path.exists(json_path):
        load_state_from_json(p, json_path)
    else:
        print(f"Warning: {json_path} not found. Running with baseline stats.")
        
    sim = CombatSimulator(p)
    result_state = sim.run_simulation()