# ==============================================================================
# Script: engine/floor_map.py
# Version: 2.0.0 (True C# Source Logic)
# Description: Generates the 24-slot environment using the exact hardcoded arrays
#              from the developer's C# source code. Eliminates Gaussian guesswork
#              in favor of top-down sequential binomial rolling.
# ==============================================================================

import os
import sys
import random

# --- BULLETPROOF PATHING ---
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

ROOT_DIR = os.path.abspath(os.path.join(SIM_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ore import Ore


class Floor:
    """Represents a single generated floor instance containing a 24-slot grid."""
    def __init__(self, floor_id, grid, is_gleaming, gleaming_multi):
        self.floor_id = floor_id
        self.grid = grid
        self.is_gleaming = is_gleaming
        self.gleaming_multi = gleaming_multi


class FloorGenerator:
    """
    Utilizes O(1) array lookups to rapidly generate floors based on the true 
    game spawn rates and tier progression metrics.
    """
    def __init__(self):
        # Maps rarity ID (0-6) to string prefix
        self.RARITY_PREFIX = {0: 'dirt', 1: 'com', 2: 'rare', 3: 'epic', 4: 'leg', 5: 'myth', 6: 'div'}

        # TIER UNLOCKS: Map Rarity ->[Tier 1, Tier 2, Tier 3, Tier 4] unlock floors
        self.TIER_UNLOCKS = {
            0:[1, 12, 24, 81],
            1:[1, 18, 30, 96],
            2:[3, 26, 36, 111],
            3:[6, 30, 42, 126],
            4: [12, 32, 45, 136],
            5: [20, 35, 50, 141],
            6:[50, 75, 100, 150]
        }

        # BOSS FLOORS (Ascension 2 Unlocked): Floor ID -> Rarity to spawn as Tier 3 Boss
        self.BOSS_FLOORS = {
            80: 0, 95: 1, 110: 2, 125: 3, 135: 4, 140: 5, 149: 6
        }

        # CHANCE SETS: (Minimum Floor, [1-in-X chances for Dirt -> Divine])
        # Formatted top-down so we easily break early on the highest applicable bracket.
        self.CHANCE_SETS =[
            (100,[3, 6, 7, 7, 7, 14, 30]),
            (70,[3, 6, 7, 7, 8, 17, 40]),
            (60,[3, 7, 7, 6, 8, 18, 45]),
            (50,[3, 7, 7, 6, 8, 18, 50]),
            (30,[3, 7, 9, 7, 8, 20, 21]),
            (25,[3, 8, 8, 7, 9, 20, 21]),
            (20,[3, 9, 9, 7, 11, 20, 21]),
            (15,[3, 9, 9, 8, 13, 20, 21]),
            (10,[3, 9, 9, 9, 14, 20, 21]),
            (5,[3, 8, 8, 10, 14, 20, 21]),
            (1,[3, 7, 9, 10, 14, 20, 21])
        ]

    def _create_ore_with_mods(self, ore_id, floor_id, player):
        """Helper to instantiate an Ore and roll its specific UI modifiers."""
        ore = Ore(ore_id, floor_id, player)
        ore.modifiers = {
            'exp_multi': player.exp_mod_gain if (random.random() < player.exp_mod_chance) else 1.0,
            'loot_multi': player.loot_mod_gain if (random.random() < player.loot_mod_chance) else 1.0,
            'stamina_gain': player.stamina_mod_gain if (random.random() < player.stamina_mod_chance) else 0.0,
            'speed_active': random.random() < player.speed_mod_chance,
            'speed_gain': player.speed_mod_gain
        }
        return ore

    def generate_floor(self, floor_id, player):
        """Builds a 24-slot floor natively matching the C# spawn arrays."""
        grid = [None] * 24

        # 1. Roll for Gleaming Floor
        is_gleaming = random.random() < player.gleaming_floor_chance
        gleaming_multi = player.gleaming_floor_multi if is_gleaming else 1.0

        # 2. Check for Ascension 2 Boss Floor Overrides
        if player.asc2_unlocked and floor_id in self.BOSS_FLOORS:
            rarity = self.BOSS_FLOORS[floor_id]
            ore_id = f"{self.RARITY_PREFIX[rarity]}3" # Bosses are always Tier 3
            
            for idx in range(24):
                grid[idx] = self._create_ore_with_mods(ore_id, floor_id, player)
            return Floor(floor_id, grid, is_gleaming, gleaming_multi)

        # 3. Find the correct Spawn Probability Bracket
        current_chances = None
        for min_f, chances in self.CHANCE_SETS:
            if floor_id >= min_f:
                current_chances = chances
                break

        # 4. Sequentially roll for all 24 slots (True C# Logic)
        for idx in range(24):
            # Roll Top-Down: Check Divine first, down to Dirt
            for rarity in range(6, -1, -1):
                
                # If this rarity isn't unlocked yet, skip
                if floor_id < self.TIER_UNLOCKS[rarity][0]:
                    continue
                    
                chance = current_chances[rarity]
                
                # Roll the 1-in-X chance
                if random.randint(1, chance) == 1:
                    # Success! Determine the Tier level for this rarity
                    tier = 1
                    if floor_id >= self.TIER_UNLOCKS[rarity][3]: tier = 4
                    elif floor_id >= self.TIER_UNLOCKS[rarity][2]: tier = 3
                    elif floor_id >= self.TIER_UNLOCKS[rarity][1]: tier = 2
                    
                    # Asc2 safety check for Tier 4s
                    if tier == 4 and not player.asc2_unlocked:
                        tier = 3
                    
                    ore_id = f"{self.RARITY_PREFIX[rarity]}{tier}"
                    grid[idx] = self._create_ore_with_mods(ore_id, floor_id, player)
                    break # Stop rolling for this slot, we found our ore!

        return Floor(floor_id, grid, is_gleaming, gleaming_multi)

# ==============================================================================
# QUICK VERIFICATION TEST
# ==============================================================================
if __name__ == "__main__":
    from core.player import Player
    
    p = Player()
    p.base_stats['Luck'] = 100
    p.set_upgrade_level(38, 20)
    p.asc2_unlocked = True

    generator = FloorGenerator()
    test_floor_id = 149
    
    print(f"--- Generating Floor {test_floor_id} ---")
    floor = generator.generate_floor(test_floor_id, p)
    
    print(f"Gleaming Floor: {floor.is_gleaming} (Multi: {floor.gleaming_multi}x)")
    
    empty_count = 0
    for i, slot in enumerate(floor.grid):
        if slot is not None:
            print(f"Slot {i}: {slot.ore_id}")
        else:
            empty_count += 1
            
    print(f"Total Empty Slots: {empty_count}")