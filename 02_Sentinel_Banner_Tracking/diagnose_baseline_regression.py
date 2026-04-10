#!/usr/bin/env python3
"""
Diagnose why specific baseline slots are regressing.

This script analyzes the frame-by-frame voting behavior to understand
why blocks that were previously correctly identified are now miscalled.
"""

import cv2
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import project_config as cfg
from ascension_detector import auto_configure_ascension

# Constants
ORE0_X, ORE0_Y = 74, 261
STEP = 59.0
SIDE_PX = 48
STATE_COMPLEXITY_THRESHOLD = 500
MIN_VOTE_CONFIDENCE = 0.30

# Setup
buffer_dir = cfg.get_buffer_path()
run_id = os.path.basename(buffer_dir).split('_')[-1]
asc_level = auto_configure_ascension()

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"final_floor_boundaries_run_{run_id}.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_dna_inventory_run_{run_id}.csv")
BULLY_PENALTIES = cfg.BULLY_PENALTIES
SKIP_IN_SHADOW = {'dirt1', 'dirt2', 'dirt3'}

def rotate_image(image, angle):
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def get_spatial_mask(r_idx):
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if r_idx == 0:
        mask[0:20, :] = 0
    return mask

def load_all_templates():
    templates = {'active': {}, 'shadow': {}}
    t_path = cfg.TEMPLATE_DIR
    
    for t_file in os.listdir(t_path):
        if not t_file.endswith('.png'):
            continue
        tier = t_file.replace('.png', '')
        img = cv2.imread(os.path.join(t_path, t_file), 0)
        if img is None or img.shape != (SIDE_PX, SIDE_PX):
            continue
        
        state = 'shadow' if '_shadow' in tier else 'active'
        tier = tier.replace('_shadow', '')
        
        if tier not in templates[state]:
            templates[state][tier] = []
        
        # Store 3 rotation variants
        for angle in [-3, 0, 3]:
            templates[state][tier].append(rotate_image(img, angle))
    
    return templates

def analyze_slot_voting(floor_id, row, slot, expected_tier):
    """
    Analyze why a specific slot is miscalled.
    
    Args:
        floor_id: Floor number
        row: Row number (1-4)
        slot: Slot number (0-5)
        expected_tier: What the baseline says it should be
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: Floor {floor_id} R{row}_S{slot} (expected: {expected_tier})")
    print(f"{'='*80}\n")
    
    # Load data
    boundaries = pd.read_csv(BOUNDARIES_CSV)
    dna_map = pd.read_csv(DNA_INVENTORY_CSV)
    templates = load_all_templates()
    
    # Get floor data
    floor_data = boundaries[boundaries['floor_id'] == floor_id]
    if floor_data.empty:
        print(f"ERROR: Floor {floor_id} not found in boundaries")
        return
    
    floor_data = floor_data.iloc[0]
    f_start = int(floor_data['true_start_frame'])
    f_end = int(floor_data['end_frame'])
    f_range = list(range(f_start, f_end + 1))
    
    # Get allowed tiers
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= floor_id <= e]
    
    # Get slot coordinates (row is 1-indexed here, convert to 0-indexed)
    r_idx = row - 1
    x1 = int(ORE0_X + slot * STEP)
    y1 = int(ORE0_Y + r_idx * STEP)
    
    print(f"Frame range: {f_start}-{f_end} ({len(f_range)} frames)")
    print(f"Slot position: ({x1}, {y1})")
    print(f"Allowed tiers: {len(allowed)}\n")
    
    # Get PNG files
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    
    # Accumulate votes with detailed logging
    tier_votes = {t: 0.0 for t in allowed}
    frame_details = []
    mask = get_spatial_mask(r_idx)
    
    # Sample first 30 frames
    sample_indices = np.linspace(0, min(len(f_range)-1, 29), min(30, len(f_range)), dtype=int)
    
    for sample_idx in sample_indices:
        f_idx = f_range[sample_idx]
        
        if f_idx >= len(all_files):
            continue
        
        fname = all_files[f_idx]
        img_path = os.path.join(buffer_dir, fname)
        img = cv2.imread(img_path, 0)  # Grayscale
        
        if img is None:
            continue
        
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX):
            continue
        
        # Determine state
        comp = cv2.Laplacian(roi, cv2.CV_64F).var()
        state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        # Score all allowed tiers
        scores = {}
        for tier in allowed:
            if tier not in templates[state]:
                continue
            
            # Skip dirt in shadow
            if state == 'shadow' and tier in SKIP_IN_SHADOW:
                continue
            
            # Get best score across rotation variants
            best_score = 0
            for tpl in templates[state][tier]:
                score = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED, mask=mask))[1]
                best_score = max(best_score, score)
            
            # Apply penalty
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            adjusted_score = best_score - penalty
            scores[tier] = adjusted_score
        
        # Get winner
        if not scores:
            continue
            
        sorted_tiers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner_tier, winner_score = sorted_tiers[0]
        runner_up_tier, runner_up_score = sorted_tiers[1] if len(sorted_tiers) > 1 else (None, 0.0)
        gap = winner_score - runner_up_score
        
        # Check if vote would be cast
        vote_cast = winner_score >= MIN_VOTE_CONFIDENCE
        
        frame_info = {
            'frame': f_idx,
            'winner': winner_tier,
            'winner_score': winner_score,
            'runner_up': runner_up_tier,
            'runner_up_score': runner_up_score,
            'gap': gap,
            'vote_cast': vote_cast,
            'expected_score': scores.get(expected_tier, 0.0),
            'expected_rank': None
        }
        
        # Find rank of expected tier
        for rank, (tier, score) in enumerate(sorted_tiers):
            if tier == expected_tier:
                frame_info['expected_rank'] = rank + 1
                break
        
        # Cast vote if above threshold
        if vote_cast:
            tier_votes[winner_tier] += winner_score
        
        frame_details.append(frame_info)
    
    # Analyze results
    print("VOTE ACCUMULATION SUMMARY:")
    print(f"{'Tier':<12} {'Votes':<10} {'Rank'}")
    print("-" * 35)
    sorted_votes = sorted(tier_votes.items(), key=lambda x: x[1], reverse=True)
    for rank, (tier, votes) in enumerate(sorted_votes[:10], 1):
        marker = " ← WINNER" if rank == 1 else ""
        marker += " ← EXPECTED" if tier == expected_tier else ""
        print(f"{tier:<12} {votes:<10.3f} #{rank}{marker}")
    
    # Find expected tier in rankings
    expected_votes = tier_votes[expected_tier]
    expected_rank = next((i+1 for i, (t, v) in enumerate(sorted_votes) if t == expected_tier), None)
    
    print(f"\nEXPECTED TIER ({expected_tier}): Rank #{expected_rank}, {expected_votes:.3f} votes")
    
    # Analyze frame patterns
    print(f"\nFRAME-BY-FRAME ANALYSIS (first 30 frames):")
    print(f"{'Frame':<8} {'Winner':<10} {'Score':<8} {'Gap':<8} {'Vote?':<6} {expected_tier+' Score':<12} {expected_tier+' Rank'}")
    print("-" * 75)
    
    votes_cast_count = 0
    votes_for_expected = 0
    votes_for_wrong = 0
    
    for info in frame_details:
        vote_str = "YES" if info['vote_cast'] else "no"
        print(f"{info['frame']:<8} {info['winner']:<10} {info['winner_score']:<8.3f} "
              f"{info['gap']:<8.3f} {vote_str:<6} {info['expected_score']:<12.3f} "
              f"#{info['expected_rank']}")
        
        if info['vote_cast']:
            votes_cast_count += 1
            if info['winner'] == expected_tier:
                votes_for_expected += 1
            else:
                votes_for_wrong += 1
    
    print(f"\nVOTING STATISTICS:")
    print(f"  Total frames analyzed: {len(frame_details)}")
    print(f"  Votes cast: {votes_cast_count}")
    print(f"  Votes for {expected_tier}: {votes_for_expected} ({100*votes_for_expected/max(votes_cast_count,1):.1f}%)")
    print(f"  Votes for wrong tier: {votes_for_wrong} ({100*votes_for_wrong/max(votes_cast_count,1):.1f}%)")
    
    # Identify the problem
    print(f"\nDIAGNOSIS:")
    
    winner_tier = sorted_votes[0][0]
    winner_votes = sorted_votes[0][1]
    
    if votes_for_expected == 0:
        print(f"  ❌ {expected_tier} NEVER won a single frame above threshold (MIN_VOTE_CONFIDENCE={MIN_VOTE_CONFIDENCE})")
        avg_expected_score = np.mean([f['expected_score'] for f in frame_details])
        print(f"     Average {expected_tier} score: {avg_expected_score:.3f}")
        if avg_expected_score < MIN_VOTE_CONFIDENCE:
            print(f"     → {expected_tier} scores are consistently below threshold")
            print(f"     → Possible causes: template mismatch, lighting variation, damage state mismatch")
    elif votes_for_expected < votes_for_wrong:
        print(f"  ⚠️  {expected_tier} won some frames ({votes_for_expected}) but {winner_tier} won more ({votes_for_wrong})")
        print(f"     → Voting accumulation is working, but wrong tier wins more often")
        print(f"     → Possible causes: template similarity, obstruction patterns, damage state confusion")
    else:
        print(f"  ⚠️  {expected_tier} won most frames but still lost overall?")
        print(f"     → Check discriminator behavior or Phase3 intervention")
    
    # Check if discriminator would apply
    if len(sorted_votes) >= 2:
        top2_gap = sorted_votes[0][1] - sorted_votes[1][1]
        if top2_gap < 0.10 * sorted_votes[0][1]:  # Within 10%
            print(f"\n  ⚙️  Discriminator likely triggered: {sorted_votes[0][0]} vs {sorted_votes[1][0]}")
            print(f"     Gap: {top2_gap:.3f} ({100*top2_gap/sorted_votes[0][1]:.1f}% of winner)")

def main():
    """Analyze key regression cases."""
    
    # Representative samples of each failure type
    test_cases = [
        # rare1 → com2 (11 cases total)
        (18, 3, 1, 'rare1'),  # Floor 18 R3_S1
        
        # epic1 → dirt1 (10 cases total)
        (7, 2, 0, 'epic1'),   # Floor 7 R2_S0
        
        # rare1 → low_conf (6 cases total)
        (21, 1, 2, 'rare1'),  # Floor 21 R1_S2
        
        # rare1 → com1 (3 cases total)
        (5, 2, 4, 'rare1'),   # Floor 5 R2_S4
    ]
    
    for floor_id, row, slot, expected in test_cases:
        analyze_slot_voting(floor_id, row, slot, expected)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
