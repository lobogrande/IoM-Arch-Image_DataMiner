#!/usr/bin/env python3
"""
Validate Block Identification Accuracy

Compares current block identification results against ground truth baseline
to measure accuracy and detect regressions.

Usage:
    python validate_block_accuracy.py
    python validate_block_accuracy.py --baseline path/to/baseline.csv --current path/to/current.csv
"""

import os
import sys
import pandas as pd
import argparse
from collections import defaultdict

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import project_config as cfg

def extract_slot_identifications(df):
    """
    Extract block identifications from CSV into a dict.
    Returns: {(floor_id, slot_name): tier_name}
    """
    identifications = {}
    
    for _, row in df.iterrows():
        floor_id = int(row['floor_id'])
        
        # Extract all slot columns (R1_S0, R2_S3, etc.)
        for col in df.columns:
            if col.startswith('R') and '_S' in col and not col.endswith('_score') and not col.endswith('_mom') and not col.endswith('_tag'):
                slot_name = col
                tier = row[col]
                
                # Skip empty/obstructed/missing values
                if pd.isna(tier) or tier in ['empty', 'obstructed', '']:
                    continue
                
                identifications[(floor_id, slot_name)] = tier
    
    return identifications

def compare_identifications(baseline, current):
    """
    Compare baseline vs current identifications.
    Returns dict with accuracy metrics.
    """
    all_slots = set(baseline.keys()) | set(current.keys())
    
    correct = 0
    incorrect = 0
    missing_in_current = 0
    extra_in_current = 0
    
    errors = []
    
    for slot_key in all_slots:
        floor_id, slot_name = slot_key
        baseline_tier = baseline.get(slot_key)
        current_tier = current.get(slot_key)
        
        if baseline_tier and current_tier:
            if baseline_tier == current_tier:
                correct += 1
            else:
                incorrect += 1
                errors.append({
                    'floor': floor_id,
                    'slot': slot_name,
                    'expected': baseline_tier,
                    'actual': current_tier
                })
        elif baseline_tier and not current_tier:
            missing_in_current += 1
            errors.append({
                'floor': floor_id,
                'slot': slot_name,
                'expected': baseline_tier,
                'actual': 'MISSING'
            })
        elif current_tier and not baseline_tier:
            extra_in_current += 1
    
    total_baseline = len(baseline)
    accuracy = (correct / total_baseline * 100) if total_baseline > 0 else 0
    
    return {
        'total_baseline_slots': total_baseline,
        'correct': correct,
        'incorrect': incorrect,
        'missing': missing_in_current,
        'extra': extra_in_current,
        'accuracy_pct': accuracy,
        'errors': errors
    }

def analyze_per_tier_accuracy(baseline, current):
    """
    Calculate precision and recall per tier.
    """
    tier_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    all_slots = set(baseline.keys()) | set(current.keys())
    
    for slot_key in all_slots:
        baseline_tier = baseline.get(slot_key)
        current_tier = current.get(slot_key)
        
        if baseline_tier == current_tier and baseline_tier:
            # True positive
            tier_stats[baseline_tier]['tp'] += 1
        elif baseline_tier and current_tier and baseline_tier != current_tier:
            # False negative for baseline tier, false positive for current tier
            tier_stats[baseline_tier]['fn'] += 1
            tier_stats[current_tier]['fp'] += 1
        elif baseline_tier and not current_tier:
            # False negative
            tier_stats[baseline_tier]['fn'] += 1
        elif current_tier and not baseline_tier:
            # False positive
            tier_stats[current_tier]['fp'] += 1
    
    # Calculate precision and recall
    tier_metrics = {}
    for tier, stats in tier_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        tier_metrics[tier] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision_pct': precision,
            'recall_pct': recall,
            'f1_score': f1
        }
    
    return tier_metrics

def main():
    parser = argparse.ArgumentParser(description='Validate block identification accuracy')
    parser.add_argument('--baseline', default=None, help='Path to baseline CSV (ground truth)')
    parser.add_argument('--current', default=None, help='Path to current results CSV')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-tier metrics')
    args = parser.parse_args()
    
    # Default paths
    RUN_ID = 0
    baseline_path = args.baseline or os.path.join(cfg.DATA_DIRS["TRACKING"], "baseline_snapshot.csv")
    current_path = args.current or os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_block_inventory_run_{RUN_ID}.csv")
    
    # Check files exist
    if not os.path.exists(baseline_path):
        print(f"ERROR: Baseline file not found: {baseline_path}")
        print("Run step6_tier_consensus.py first to generate results.")
        return 1
    
    if not os.path.exists(current_path):
        print(f"ERROR: Current results file not found: {current_path}")
        print("Run step6_tier_consensus.py first to generate results.")
        return 1
    
    print(f"Loading baseline: {os.path.basename(baseline_path)}")
    baseline_df = pd.read_csv(baseline_path)
    baseline = extract_slot_identifications(baseline_df)
    
    print(f"Loading current:  {os.path.basename(current_path)}")
    current_df = pd.read_csv(current_path)
    current = extract_slot_identifications(current_df)
    
    print("\n" + "="*80)
    print("ACCURACY REPORT")
    print("="*80)
    
    # Overall accuracy
    results = compare_identifications(baseline, current)
    print(f"\nTotal slots in baseline: {results['total_baseline_slots']}")
    print(f"Correct identifications:  {results['correct']}")
    print(f"Incorrect identifications: {results['incorrect']}")
    print(f"Missing in current:       {results['missing']}")
    print(f"Extra in current:         {results['extra']}")
    print(f"\nOVERALL ACCURACY: {results['accuracy_pct']:.2f}%")
    
    # Show errors if any
    if results['errors']:
        print(f"\n{'='*80}")
        print(f"ERRORS ({len(results['errors'])} total)")
        print("="*80)
        for err in results['errors'][:20]:  # Show first 20
            print(f"  Floor {err['floor']} {err['slot']}: Expected {err['expected']}, Got {err['actual']}")
        if len(results['errors']) > 20:
            print(f"  ... and {len(results['errors']) - 20} more errors")
    
    # Per-tier metrics
    if args.detailed:
        print(f"\n{'='*80}")
        print("PER-TIER METRICS")
        print("="*80)
        tier_metrics = analyze_per_tier_accuracy(baseline, current)
        
        # Sort by tier name
        for tier in sorted(tier_metrics.keys()):
            metrics = tier_metrics[tier]
            print(f"\n{tier}:")
            print(f"  Precision: {metrics['precision_pct']:.2f}%")
            print(f"  Recall:    {metrics['recall_pct']:.2f}%")
            print(f"  F1 Score:  {metrics['f1_score']:.2f}")
            print(f"  (TP={metrics['true_positives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']})")
    
    # Return exit code based on accuracy
    if results['accuracy_pct'] >= 99.5:
        print(f"\n✓ Accuracy is excellent ({results['accuracy_pct']:.2f}%)")
        return 0
    elif results['accuracy_pct'] >= 95.0:
        print(f"\n⚠ Accuracy is acceptable ({results['accuracy_pct']:.2f}%) but could be improved")
        return 0
    else:
        print(f"\n✗ Accuracy is below acceptable threshold ({results['accuracy_pct']:.2f}%)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
