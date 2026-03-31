# ==============================================================================
# Script: core/skills.py
# Version: 1.0.0 (Modular Architecture)
# Description: State-tracker for Enrage, Flurry, and Quake. Handles continuous 
#              cooldown timers, active duration timers, charge consumption, 
#              auto-casting, and the Ability Instacharge RNG loops.
# ==============================================================================

import random

class SkillManager:
    def __init__(self, player):
        """
        Initializes the skill trackers. 
        Takes the Player object to read max cooldowns, durations, and charges.
        """
        self.player = player
        
        # Current State Timers & Counters
        self.enrage_cd = 0.0
        self.enrage_charges = 0
        
        self.flurry_cd = 0.0
        self.flurry_timer = 0.0
        
        self.quake_cd = 0.0
        self.quake_charges = 0
        
        # Lifetime stats (Useful for optimizers/analytics later)
        self.total_enrage_casts = 0
        self.total_flurry_casts = 0
        self.total_quake_casts = 0
        self.total_instacharges = 0

    def tick(self, dt):
        """
        Advances time by 'dt' (delta time) seconds.
        Handles cooldowns, Flurry's duration, and Auto-Casting.
        Returns a dictionary of events (e.g., flat stamina restored by Flurry).
        """
        events = {"stamina_restored": 0.0}
        
        # 1. Advance all active timers
        if self.enrage_cd > 0: self.enrage_cd -= dt
        if self.flurry_cd > 0: self.flurry_cd -= dt
        if self.quake_cd > 0: self.quake_cd -= dt
        
        if self.flurry_timer > 0: 
            self.flurry_timer -= dt
            if self.flurry_timer < 0: 
                self.flurry_timer = 0.0

        insta_chance = self.player.ability_insta_charge
        
        # 2. Auto-Cast: Enrage (Requires Upgrade 8 >= 1)
        chain = 0
        if self.player.upgrade_levels.get(8, 0) >= 1:
            while self.enrage_cd <= 0 and chain < 100:
                self.enrage_charges += self.player.enrage_charges # FIX: += instead of =
                self.enrage_cd = self.player.enrage_cooldown
                self.total_enrage_casts += 1
                
                # Roll for Ability Instacharge (Independent probabilities naturally cascade/decrease)
                if random.random() < insta_chance:
                    self.enrage_cd = 0.0
                    self.total_instacharges += 1
                    chain += 1
                else:
                    break

        # 3. Auto-Cast: Flurry (Requires Upgrade 8 >= 2)
        chain = 0
        if self.player.upgrade_levels.get(8, 0) >= 2:
            while self.flurry_cd <= 0 and chain < 100:
                self.flurry_timer += self.player.flurry_duration # FIX: += instead of =
                self.flurry_cd = self.player.flurry_cooldown
                self.total_flurry_casts += 1
                
                # Flurry grants flat stamina on cast
                events["stamina_restored"] += self.player.flurry_sta_on_cast
                
                if random.random() < insta_chance:
                    self.flurry_cd = 0.0
                    self.total_instacharges += 1
                    chain += 1
                else:
                    break

        # 4. Auto-Cast: Quake (Requires Upgrade 8 >= 3)
        chain = 0
        if self.player.upgrade_levels.get(8, 0) >= 3:
            while self.quake_cd <= 0 and chain < 100:
                self.quake_charges += self.player.quake_attacks # FIX: += instead of =
                self.quake_cd = self.player.quake_cooldown
                self.total_quake_casts += 1
                
                if random.random() < insta_chance:
                    self.quake_cd = 0.0
                    self.total_instacharges += 1
                    chain += 1
                else:
                    break

        return events

    def consume_attack(self):
        """
        Called by the Combat Loop every time the player lands a hit.
        Decrements active skill charges.
        Returns True if a Quake AoE splash attack should trigger.
        """
        quake_triggered = False
        
        if self.enrage_charges > 0:
            self.enrage_charges -= 1
            
        if self.quake_charges > 0:
            self.quake_charges -= 1
            quake_triggered = True
            
        return quake_triggered

    # --------------------------------------------------------------------------
    # ACTIVE STATE PROPERTIES (Polled by the Combat Loop)
    # --------------------------------------------------------------------------
    @property
    def is_enrage_active(self):
        return self.enrage_charges > 0
        
    @property
    def is_flurry_active(self):
        return self.flurry_timer > 0

    @property
    def is_quake_active(self):
        return self.quake_charges > 0