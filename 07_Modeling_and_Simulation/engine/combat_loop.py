# ==============================================================================
# Script: engine/combat_loop.py
# Version: 1.2.0 (Modular Architecture - High-Performance Cached Edition)
# Description: The core simulation engine. Executes a run floor-by-floor using 
#              micro-tick hit-by-hit combat. Features extreme loop-hoisting to
#              minimize @property accesses and maximize Monte Carlo throughput.
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
        self.specific_ores_mined = {} # <--- Specific tracking for Card Farming
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

    def _process_kill_rewards(self, ore, floor_obj, state: RunState, p_max_sta):
        """Processes loot, XP, and modifiers when an ore HP hits 0."""
        xp_yield = ore.xp * ore.modifiers.get('exp_multi', 1.0) * floor_obj.gleaming_multi
        state.total_xp += xp_yield
        
        loot_yield = ore.frag_amt * ore.modifiers.get('loot_multi', 1.0) * floor_obj.gleaming_multi
        if ore.frag_type in state.total_frags:
            state.total_frags[ore.frag_type] += loot_yield
            
        sta_gain = ore.modifiers.get('stamina_gain', 0.0)
        if sta_gain > 0:
            state.stamina = min(p_max_sta, state.stamina + sta_gain)
            
        if ore.modifiers.get('speed_active', False):
            state.speed_pool += ore.modifiers.get('speed_gain', 0.0)
            
        state.ores_mined += 1
        
        # Tracking specific ore tier/type kills
        ore_id = ore.ore_id
        state.specific_ores_mined[ore_id] = state.specific_ores_mined.get(ore_id, 0) + 1


    def run_simulation(self):
        # ======================================================================
        # LOOP HOISTING (ATTRIBUTE CACHING)
        # We cache static @property calculations into local variables once per 
        # run to prevent billions of redundant dictionary lookups during the loop.
        # ======================================================================
        p_max_sta = self.player.max_sta
        p_atk_spd = self.player.atk_spd
        p_speed_mod_atk_rate = self.player.speed_mod_attack_rate
        p_flurry_bonus_atk_spd = self.player.flurry_bonus_atk_spd
        p_damage = self.player.damage
        p_armor_pen = self.player.armor_pen
        p_enrage_bonus_dmg = self.player.enrage_bonus_dmg
        p_quake_dmg_to_all = self.player.quake_dmg_to_all
        
        # Crit Cache
        p_u_crit_ch = self.player.ultra_crit_chance
        p_u_crit_dmg = self.player.ultra_crit_dmg_mult
        p_s_crit_ch = self.player.super_crit_chance
        p_s_crit_dmg = self.player.super_crit_dmg_mult
        p_crit_ch = self.player.crit_chance
        p_crit_dmg = self.player.crit_dmg_mult
        p_enrage_bonus_crit_dmg = self.player.enrage_bonus_crit_dmg

        # Using a fast local closure instead of a class method eliminates
        # function-call overhead and 'self.' lookups on every single hit.
        def roll_crit(is_enrage_active):
            rand_val = random.random()
            if rand_val < p_u_crit_ch: 
                return p_u_crit_dmg, 'ultra'
            if rand_val < p_s_crit_ch: 
                return p_s_crit_dmg, 'super'
            if rand_val < p_crit_ch:
                bonus = p_enrage_bonus_crit_dmg if is_enrage_active else 0.0
                return p_crit_dmg + bonus, 'crit'
            return 1.0, 'normal'

        # ======================================================================
        
        state = RunState(self.player)
        skills = SkillManager(self.player)
        current_floor_id = 1
        
        print("\n[ SIMULATION STARTED ]")
        state.record_telemetry()
        
        while state.stamina > 0:
            floor = self.generator.generate_floor(current_floor_id, self.player)
            state.highest_floor = current_floor_id
            
            for i, slot_idx in enumerate(PATH_ORDER):
                if state.stamina <= 0: break
                    
                target_ore = floor.grid[slot_idx]
                if target_ore is None or target_ore.hp <= 0: continue
                    
                state.stamina -= STAMINA_COST_PER_ORE
                
                # --- MICRO-TICK COMBAT LOOP ---
                while target_ore.hp > 0 and state.stamina > 0:
                    
                    is_flurry = skills.is_flurry_active
                    is_enrage = skills.is_enrage_active
                    
                    flurry_mult = 1.0 + p_flurry_bonus_atk_spd if is_flurry else 1.0
                    
                    if state.speed_pool > 0:
                        current_atk_spd = p_atk_spd * p_speed_mod_atk_rate * flurry_mult
                        state.speed_pool -= 1
                    else:
                        current_atk_spd = p_atk_spd * flurry_mult
                        
                    time_passed = 1.0 / current_atk_spd
                    state.total_time += time_passed
                    
                    events = skills.tick(time_passed)
                    if events["stamina_restored"] > 0:
                        state.stamina = min(p_max_sta, state.stamina + events["stamina_restored"])
                        
                    crit_mult, crit_type = roll_crit(is_enrage)
                    state.hit_counts[crit_type] += 1
                    
                    base_dmg = p_damage
                    if is_enrage: 
                        base_dmg *= (1.0 + p_enrage_bonus_dmg)
                        
                    # Safely calculate effective armor (can't go below 0)
                    eff_armor = max(0, target_ore.armor - p_armor_pen)
                    
                    actual_dmg = max(1.0, base_dmg - eff_armor) * crit_mult
                    target_ore.hp -= actual_dmg
                    state.stamina -= STAMINA_COST_PER_HIT
                    
                    # Quake AOE Proc
                    if skills.consume_attack():
                        q_base = p_damage * p_quake_dmg_to_all
                        for bg_idx in PATH_ORDER[i+1:]:
                            bg_ore = floor.grid[bg_idx]
                            if bg_ore is not None and bg_ore.hp > 0:
                                q_crit, q_type = roll_crit(is_enrage)
                                state.hit_counts[q_type] += 1
                                
                                bg_eff_armor = max(0, bg_ore.armor - p_armor_pen)
                                q_dmg = max(1.0, q_base - bg_eff_armor) * q_crit
                                bg_ore.hp -= q_dmg
                                if bg_ore.hp <= 0:
                                    self._process_kill_rewards(bg_ore, floor, state, p_max_sta)
                                    
                if target_ore.hp <= 0:
                    self._process_kill_rewards(target_ore, floor, state, p_max_sta)
                
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