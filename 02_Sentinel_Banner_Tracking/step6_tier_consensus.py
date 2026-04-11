# step6_tier_consensus.py
# Purpose: Master Plan Step 6 - Production Run for all floors (Dynamic bounds).
# Version: 7.1 (Per-frame two-pass: if ANY frame has dirt<0.05 from higher tier, override)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg
from ascension_detector import auto_configure_ascension
from pairwise_discriminators import should_use_pairwise_discriminator, apply_pairwise_discriminator

# CRITICAL: Configure ascension BEFORE any parallel workers are spawned
detected_asc = auto_configure_ascension(verbose=False)  # Suppress worker output
print(f"[MODULE INIT] Detected: {detected_asc}, BOSS_DATA has {len(cfg.BOSS_DATA)} floors: {list(cfg.BOSS_DATA.keys())}")

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"final_floor_boundaries_run_{RUN_ID}.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_dna_inventory_run_{RUN_ID}.csv")
HOMING_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{RUN_ID}.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_block_inventory_run_{RUN_ID}.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], f"block_identification_proofs_run_{RUN_ID}")

# --- 1. VALIDATED PIXEL CONSTANTS ---
ORE0_X, ORE0_Y = 74, 261 
STEP = 59.0
SIDE_PX = 48 
HUD_DX, HUD_DY = 20, 30

# --- 2. PRODUCTION CONTROLS ---
# Set to None to process ALL floors in the dataset, or set integers for specific diagnostic ranges
START_FLOOR_BOUND = None    
END_FLOOR_BOUND = None      
MAX_SAMPLES = 10  # Reduced from 40 to focus on pristine blocks before damage         
# Surgical Gate: Raised from 0.30 to 0.40 for better confidence
MIN_VOTE_CONFIDENCE = 0.30  # Lowered from 0.40 to allow com2 blocks (score ~0.33-0.35) to accumulate votes 
STATE_COMPLEXITY_THRESHOLD = 500
ROTATION_VARIANTS = [-3, 0, 3]

# BULLY SHIELD: Now loaded from project_config based on ascension level
# ASC1: Original penalties (validated to work)
# ASC2: Aggressive penalties based on template similarity analysis
# Will be set to cfg.BULLY_PENALTIES after ascension detection
BULLY_PENALTIES = cfg.BULLY_PENALTIES  # Initialized from config

# DIRT SKIP: Dirt blocks die in 1 hit, rarely damaged/shadowed
# Skip dirt templates in shadow matching to prevent false positives
SKIP_IN_SHADOW = {'dirt1', 'dirt2', 'dirt3'}

# Tier groups for special handling
DIRT_TIERS = {'dirt1', 'dirt2', 'dirt3', 'dirt4'}
LEG_TIERS = {'leg1', 'leg2', 'leg3', 'leg4'}
MYTH_TIERS = {'myth1', 'myth2', 'myth3', 'myth4'}

def rotate_image(image, angle):
    if angle == 0: return image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def weighted_template_match(roi, template, weighted_mask):
    """
    Custom template matching that respects pixel weights.
    
    Standard cv2.matchTemplate treats masks as binary.
    This implementation weights pixels by their discriminative importance.
    
    Args:
        roi: Region of interest (H, W)
        template: Template (H, W)
        weighted_mask: Mask with weights (0-255), where 255=important, 25=less important
    
    Returns:
        Weighted correlation score (0-1 range, similar to TM_CCOEFF_NORMED)
    """
    # Normalize mask weights to 0-1
    mask_weights = weighted_mask.astype(float) / 255.0
    
    # Only compute where mask is non-zero
    valid_pixels = mask_weights > 0
    
    if not valid_pixels.any():
        return 0.0
    
    # Extract valid pixels with weights
    roi_pixels = roi[valid_pixels].astype(float)
    tpl_pixels = template[valid_pixels].astype(float)
    weights = mask_weights[valid_pixels]
    
    # Weighted mean
    roi_mean = np.average(roi_pixels, weights=weights)
    tpl_mean = np.average(tpl_pixels, weights=weights)
    
    # Weighted standard deviation
    roi_var = np.average((roi_pixels - roi_mean)**2, weights=weights)
    tpl_var = np.average((tpl_pixels - tpl_mean)**2, weights=weights)
    
    roi_std = np.sqrt(roi_var) if roi_var > 0 else 1.0
    tpl_std = np.sqrt(tpl_var) if tpl_var > 0 else 1.0
    
    # Weighted correlation coefficient
    numerator = np.average((roi_pixels - roi_mean) * (tpl_pixels - tpl_mean), weights=weights)
    denominator = roi_std * tpl_std
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    
    # Clip to [-1, 1] range and shift to [0, 1]
    correlation = np.clip(correlation, -1, 1)
    normalized_score = (correlation + 1) / 2  # Convert to 0-1 range
    
    return normalized_score

def get_spatial_mask(r_idx, slot_id, homing_slot):
    """
    Create adaptive spatial mask based on row and player position.
    
    Args:
        r_idx: Row index (0-3)
        slot_id: Current slot ID (0-23)
        homing_slot: Player's current slot ID (or -99 if not found)
    
    Returns:
        Mask with 255=use, 0=ignore
    """
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    
    # Base circular mask - focus on center 18px radius
    cv2.circle(mask, (24, 24), 18, 255, -1)
    
    # Row 1 special: Mask top 20px to avoid dig stage text
    if r_idx == 0:
        mask[0:20, :] = 0
    
    # Mask top 8px for ALL rows to avoid modifiers
    # (modifiers appear above blocks randomly)
    mask[0:8, :] = 0
    
    # Player-adjacent masking: If player is to left or right, mask that side
    # Rows 1&3 (0,2): Player typically left, mask left side
    # Rows 2&4 (1,3): Player typically right, mask right side
    if homing_slot != -99 and abs(homing_slot - slot_id) == 1:
        # Player is adjacent (1 slot away horizontally)
        if homing_slot < slot_id:
            # Player is to the LEFT, mask left 12px
            mask[:, 0:12] = 0
        else:
            # Player is to the RIGHT, mask right 12px
            mask[:, 36:48] = 0
    
    return mask

def load_all_templates():
    templates = {'active': {}, 'shadow': {}}
    t_path = cfg.TEMPLATE_DIR
    print(f"Loading Resources from {t_path}...")
    
    # Track template counts
    template_counts = {'active': {}, 'shadow': {}}
    
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "negative", "player"]): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: 
            templates[state][tier] = []
            template_counts[state][tier] = 0
        img_native = cv2.resize(img_raw, (SIDE_PX, SIDE_PX))
        template_counts[state][tier] += 1
        for angle in ROTATION_VARIANTS:
            templates[state][tier].append(rotate_image(img_native, angle))
    
    # Print template counts
    print(f"\n[Template Summary]")
    print(f"Active templates: {sorted(template_counts['active'].items())}")
    print(f"Shadow templates: {sorted(template_counts['shadow'].items())}")
    
    # Highlight tiers with multiple templates
    multi_template_tiers = [tier for tier, count in template_counts['active'].items() if count > 1]
    if multi_template_tiers:
        print(f"[Multi-Template Tiers] Active: {sorted(multi_template_tiers)}")
    
    print()
    return templates

def check_side_slice_forensics(roi_gray):
    left_slice = roi_gray[15:40, 1:3]
    right_slice = roi_gray[15:40, 45:47]
    best_std = min(np.std(left_slice), np.std(right_slice))
    return best_std, best_std < 11.5

def get_overlap_slot(homing_id):
    if homing_id is None or homing_id < 0: return -99
    row = homing_id // 6
    
    # Directional Logic: Left-facing player (Row 2 / Slots 6-11) overlaps the slot to their right (+1)
    # Right-facing player (Row 1 / Slots 0-5) overlaps the slot to their left (-1)
    is_facing_left = (homing_id >= 6)
    overlap_candidate = (homing_id + 1) if is_facing_left else (homing_id - 1)
    
    if overlap_candidate // 6 != row or not (0 <= overlap_candidate <= 23):
        return -99
    return overlap_candidate

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map, f_id):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2)
    slot_id = r_idx * 6 + col_idx
    
    # Debug flags for problematic slots
    DEBUG_THIS_SLOT = False
    # if (f_id == 59 and r_idx == 0 and col_idx == 1) or \
    #    (f_id == 60 and r_idx == 0 and col_idx == 0) or \
    #    (f_id == 60 and r_idx == 0 and col_idx == 3) or \
    #    (f_id == 62 and r_idx == 1 and col_idx == 5):
    #     DEBUG_THIS_SLOT = True
    #     print(f"\n[DEBUG-START] F{f_id} R{r_idx+1}_S{col_idx}", flush=True)

    sample_indices = f_range[:MAX_SAMPLES]
    votes = defaultdict(float)
    tier_totals = defaultdict(float)  # Track ALL tier scores for Phase3 rescue
    tier_frame_counts = defaultdict(int)  # Count frames where each tier was scored
    tier_best_frames = defaultdict(lambda: {'frame_idx': None, 'score': 0.0})  # Track best frame for each tier
    frames_obstructed = 0
    clean_frames_processed = 0
    obstructed_sample_roi = None
    extended_sampling = False  # Track if we've already extended

    # Per-frame tracking for two-pass override
    frame_winners = []  # List of (dirt_tier, higher_tier, gap) for close calls

    for f_idx in sample_indices:
        homing_slot = get_overlap_slot(homing_map.get(f_idx))
        player_is_blocking = (homing_slot == slot_id)
        
        # Create adaptive mask based on player position
        base_mask = get_spatial_mask(r_idx, slot_id, homing_slot)
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)  # Grayscale for template matching
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX): continue
        
        # Also load in color for pairwise discriminators (if needed)
        img_color = None  # Load lazily only if needed

        if player_is_blocking:
            frames_obstructed += 1
            if obstructed_sample_roi is None: obstructed_sample_roi = roi.copy()
            if DEBUG_THIS_SLOT:
                print(f"  Frame {f_idx}: BLOCKED by player (slot_id={slot_id}, homing_slot={homing_slot})", flush=True)
            continue

        comp = cv2.Laplacian(roi, cv2.CV_64F).var()
        target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        # Score all tiers for this frame
        frame_scores = {}
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            
            # Skip dirt in shadow state (dirt dies in 1 hit, rarely damaged)
            if target_state == 'shadow' and tier in SKIP_IN_SHADOW:
                continue
            
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            bias = 0.0
            
            best_tier_score = 0
            for tpl in res[target_state][tier]:
                score = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED, mask=base_mask))[1]
                total_score = score - penalty + bias
                best_tier_score = max(best_tier_score, total_score)
            
            if best_tier_score > 0:
                frame_scores[tier] = best_tier_score
        
        # Find winner for this frame
        if frame_scores:
            sorted_frame_tiers = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
            best_f_tier = sorted_frame_tiers[0][0]
            best_f_score = sorted_frame_tiers[0][1]
            
            # Track ALL tier scores for Phase3 rescue (even if below MIN_VOTE_CONFIDENCE)
            for tier, score in frame_scores.items():
                tier_totals[tier] += score
                tier_frame_counts[tier] += 1
                # Track best frame for each tier (for discriminators)
                if score > tier_best_frames[tier]['score']:
                    tier_best_frames[tier] = {'frame_idx': f_idx, 'score': score}
            
            # Determine tier-specific vote threshold
            # com3 needs lower threshold due to template quality
            tier_vote_threshold = MIN_VOTE_CONFIDENCE
            if best_f_tier in ['com3', 'com4']:
                tier_vote_threshold = 0.25  # F59: com3=0.268
            
            if best_f_score > tier_vote_threshold:
                votes[best_f_tier] += best_f_score
                clean_frames_processed += 1
                
                if DEBUG_THIS_SLOT:
                    print(f"  Frame {f_idx}: winner={best_f_tier} score={best_f_score:.4f} (passed tier_threshold={tier_vote_threshold})", flush=True)
                
                # Check if this frame has dirt beating higher tier by <0.05
                if best_f_tier in DIRT_TIERS:
                    for tier, score in frame_scores.items():
                        if tier not in DIRT_TIERS and best_f_score - score <= 0.05:
                            frame_winners.append((best_f_tier, tier, best_f_score - score))
                            break
            else:
                if DEBUG_THIS_SLOT:
                    print(f"  Frame {f_idx}: winner={best_f_tier} score={best_f_score:.4f} (BELOW tier_threshold={tier_vote_threshold}, not counted)", flush=True)

    if DEBUG_THIS_SLOT:
        print(f"[DEBUG-VOTES] votes={dict(votes)}, clean_frames={clean_frames_processed}, obstructed={frames_obstructed}/{len(sample_indices)}", flush=True)
    
    # SMART FRAME EXTENSION: If we have very few clean frames (<3) and haven't extended yet,
    # try sampling more frames from later in the floor to get past fairy/crosshair overlaps
    if clean_frames_processed < 3 and not extended_sampling and len(f_range) > MAX_SAMPLES:
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-EXTEND] Only {clean_frames_processed} clean frames, extending sample range from {len(f_range)} total floor frames...", flush=True)
        
        extended_sampling = True
        # Sample up to 20 more frames (total 30), but don't exceed floor boundaries
        end_idx = min(MAX_SAMPLES + 20, len(f_range))
        extended_indices = f_range[MAX_SAMPLES:end_idx]
        
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-EXTEND] Sampling {len(extended_indices)} additional frames (indices {MAX_SAMPLES} to {end_idx-1})", flush=True)
        
        for f_idx in extended_indices:
            homing_slot = get_overlap_slot(homing_map.get(f_idx))
            player_is_blocking = (homing_slot == slot_id)
            
            base_mask = get_spatial_mask(r_idx, slot_id, homing_slot)
            
            img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
            if img is None: continue
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            if roi.shape != (SIDE_PX, SIDE_PX): continue

            if player_is_blocking:
                frames_obstructed += 1
                if obstructed_sample_roi is None: obstructed_sample_roi = roi.copy()
                if DEBUG_THIS_SLOT:
                    print(f"  Extended Frame {f_idx}: BLOCKED by player", flush=True)
                continue

            comp = cv2.Laplacian(roi, cv2.CV_64F).var()
            target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
            
            frame_scores = {}
            for tier in allowed_tiers:
                if tier not in res[target_state]: continue
                if target_state == 'shadow' and tier in SKIP_IN_SHADOW:
                    continue
                
                penalty = BULLY_PENALTIES.get(tier, 0.0)
                bias = 0.0
                
                best_tier_score = 0
                for tpl in res[target_state][tier]:
                    score = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED, mask=base_mask))[1]
                    total_score = score - penalty + bias
                    best_tier_score = max(best_tier_score, total_score)
                
                if best_tier_score > 0:
                    frame_scores[tier] = best_tier_score
            
            if frame_scores:
                sorted_frame_tiers = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
                best_f_tier = sorted_frame_tiers[0][0]
                best_f_score = sorted_frame_tiers[0][1]
                
                # Determine tier-specific vote threshold (same as main sampling)
                tier_vote_threshold = MIN_VOTE_CONFIDENCE
                if best_f_tier in ['com3', 'com4']:
                    tier_vote_threshold = 0.25
                
                if best_f_score > tier_vote_threshold:
                    votes[best_f_tier] += best_f_score
                    clean_frames_processed += 1
                    
                    if DEBUG_THIS_SLOT:
                        print(f"  Extended Frame {f_idx}: winner={best_f_tier} score={best_f_score:.4f} (passed tier_threshold={tier_vote_threshold})", flush=True)
                    
                    # Also credit close runner-ups
                    if len(sorted_frame_tiers) >= 2:
                        runner_up_tier = sorted_frame_tiers[1][0]
                        runner_up_score = sorted_frame_tiers[1][1]
                        frame_gap = best_f_score - runner_up_score
                        if frame_gap < 0.10:
                            votes[runner_up_tier] += runner_up_score
                    
                    # Track dirt vs higher tier close calls
                    if best_f_tier in DIRT_TIERS:
                        for tier, score in frame_scores.items():
                            if tier not in DIRT_TIERS and best_f_score - score <= 0.05:
                                frame_winners.append((best_f_tier, tier, best_f_score - score))
                                break
                else:
                    if DEBUG_THIS_SLOT:
                        print(f"  Extended Frame {f_idx}: winner={best_f_tier} score={best_f_score:.4f} (BELOW tier_threshold={tier_vote_threshold}, not counted)", flush=True)
            
            # Stop extending if we have enough good frames
            if clean_frames_processed >= 5:
                if DEBUG_THIS_SLOT:
                    print(f"[DEBUG-EXTEND] Reached {clean_frames_processed} clean frames, stopping extension", flush=True)
                break
        
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-EXTEND] After extension: votes={dict(votes)}, clean_frames={clean_frames_processed}", flush=True)
    
    if frames_obstructed / (len(sample_indices) + (20 if extended_sampling else 0)) >= 0.90 and obstructed_sample_roi is not None:
        val, is_empty = check_side_slice_forensics(obstructed_sample_roi)
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-OBSTRUCTED] 90%+ obstructed, forensics: is_empty={is_empty}, val={val:.4f}", flush=True)
        if is_empty: return "likely_empty", round(val, 4), frames_obstructed, "[L]"
        else: return "obstructed", 0, frames_obstructed, "[O]"

    if clean_frames_processed > 0:
        winner = max(votes, key=votes.get)
        winner_score = votes[winner] / clean_frames_processed
        
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-WINNER] winner={winner}, score={winner_score:.4f}", flush=True)
        
        # Get runner-up for pairwise discrimination
        sorted_candidates = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_candidates) >= 2:
            runner_up = sorted_candidates[1][0]
            runner_up_score = sorted_candidates[1][1] / clean_frames_processed
            gap = winner_score - runner_up_score
            
            # PHASE 2: Pairwise Discriminator (if scores are close)
            # Increased threshold from 0.05 to 0.10 to catch more confusable cases
            GAP_THRESHOLD = 0.10
            
            if gap < GAP_THRESHOLD:
                if should_use_pairwise_discriminator(winner, runner_up):
                    # Load first frame in color for discriminator
                    first_frame_idx = sample_indices[0]
                    img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                    if img_color is not None:
                        roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                        if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                            override_winner = apply_pairwise_discriminator(winner, runner_up, roi_color)
                            
                            if override_winner and override_winner != winner:
                                # Debug: Track when discriminators override
                                print(f"[Phase2] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap:.3f})", flush=True)
                                winner = override_winner
                                # Update score to match new winner
                                winner_score = votes.get(override_winner, 0) / clean_frames_processed
                                # Phase2 does NOT bypass threshold - Phase3 checks still need to run
        
        # PER-FRAME TWO-PASS: DISABLED - was causing false negatives
        # Original logic: If ANY frame had dirt beating higher tier by <0.05, override to higher tier
        # Problem: This was overriding legitimate dirt wins (F28 R3 S3: dirt3=0.334 → epic1=0.192)
        # Keeping the frame_winners collection for potential future use, but not applying override
        # if winner in DIRT_TIERS and frame_winners:
        #     higher_tier_candidates = defaultdict(int)
        #     for _, higher_tier, _ in frame_winners:
        #         higher_tier_candidates[higher_tier] += 1
        #     if higher_tier_candidates:
        #         override_tier = max(higher_tier_candidates, key=higher_tier_candidates.get)
        #         winner = override_tier
        #         winner_score = votes.get(override_tier, 0) / clean_frames_processed
        
        # REVERSE CHECK: If winner is NOT dirt, but dirt won multiple individual frames strongly
        # This handles cases where dirt2 dominates some frames but loses due to consistency bias
        if winner not in DIRT_TIERS:
            # Check if any dirt tier has strong frame wins
            for dirt_tier in DIRT_TIERS:
                if dirt_tier in votes:
                    dirt_vote_count = votes[dirt_tier]
                    dirt_avg_score = dirt_vote_count / clean_frames_processed
                    gap_to_winner = winner_score - dirt_avg_score
                    
                    # If dirt tier is close (<0.25) and was in allowed tiers, run discriminator
                    # Increased from 0.20 to 0.25 because dirt blocks actually match well (94%+)
                    # but lose to similar tiers due to penalties. Need wider net to catch them.
                    if gap_to_winner < 0.25 and dirt_tier in allowed_tiers:
                        if should_use_pairwise_discriminator(winner, dirt_tier):
                            # Load first frame in color for discriminator
                            first_frame_idx = sample_indices[0]
                            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                            if img_color is not None:
                                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                                    override_winner = apply_pairwise_discriminator(winner, dirt_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        print(f"[Phase3-Dirt] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap_to_winner:.3f})", flush=True)
                                        winner = override_winner
                                        winner_score = dirt_avg_score
                                        # Discriminator overrides are high-confidence - bypass threshold check
                                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"
                                    break  # Only check first close dirt tier
        
        # PHASE 3-LEG: Check if leg tier lost but should have won
        # Similar to Phase3-Dirt, handles cases where leg1 templates don't match damaged blocks well
        # but epic1 templates score even better on damaged leg1 blocks
        if winner not in LEG_TIERS:
            for leg_tier in LEG_TIERS:
                if leg_tier in votes:
                    leg_vote_count = votes[leg_tier]
                    leg_avg_score = leg_vote_count / clean_frames_processed
                    gap_to_winner = winner_score - leg_avg_score
                    
                    # If leg tier is close (<0.20) and was in allowed tiers, run discriminator
                    # leg1 vs epic1 is a known problematic pair due to template quality
                    # Increased from 0.15 to 0.20 to catch more cases where leg1 dominated early frames
                    if gap_to_winner < 0.20 and leg_tier in allowed_tiers:
                        if should_use_pairwise_discriminator(winner, leg_tier):
                            # Load first frame in color for discriminator
                            first_frame_idx = sample_indices[0]
                            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                            if img_color is not None:
                                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                                    override_winner = apply_pairwise_discriminator(winner, leg_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        print(f"[Phase3-Leg] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap_to_winner:.3f})", flush=True)
                                        winner = override_winner
                                        winner_score = leg_avg_score
                                        # Discriminator overrides are high-confidence - bypass threshold check
                                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"
                                    break  # Only check first close leg tier
        
        # PHASE 3-EPIC: Check if epic1 lost to dirt1 but should have won
        # dirt1 templates sometimes match epic1 blocks better than epic1 templates (template quality issue)
        # Similar to Phase3-Dirt and Phase3-Leg
        if winner in DIRT_TIERS:
            # Check if epic1 was a candidate and is close
            for epic_tier in ['epic1', 'epic2', 'epic3', 'epic4']:
                # Use tier_totals instead of votes - epic may have scored but never won a frame
                if epic_tier in tier_totals and epic_tier in allowed_tiers:
                    epic_avg_score = tier_totals[epic_tier] / tier_frame_counts[epic_tier]
                    gap_to_winner = winner_score - epic_avg_score
                    
                    # If epic tier is close (<0.20) and was in allowed tiers, run discriminator
                    # Increased threshold to catch cases where dirt1 matches epic1 well (F10: gap=0.14)
                    if gap_to_winner < 0.20:
                        if should_use_pairwise_discriminator(winner, epic_tier):
                            # Load first frame in color for discriminator
                            first_frame_idx = sample_indices[0]
                            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                            if img_color is not None:
                                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                                    override_winner = apply_pairwise_discriminator(winner, epic_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        # Check if com tier is also close - may be com instead of epic
                                        # F48 R1_S2: dirt3→epic3 but actually com3
                                        # Use best frame where com3 scored well for accurate discrimination
                                        com_tier = epic_tier.replace('epic', 'com')  # epic3 → com3
                                        if DEBUG_THIS_SLOT:
                                            print(f"[DEBUG] Checking {com_tier}: in tier_totals={com_tier in tier_totals}, in allowed={com_tier in allowed_tiers}", flush=True)
                                        if com_tier in tier_totals and com_tier in allowed_tiers:
                                            com_avg_score = tier_totals[com_tier] / tier_frame_counts[com_tier]
                                            gap_epic_to_com = abs(epic_avg_score - com_avg_score)
                                            if DEBUG_THIS_SLOT:
                                                print(f"[DEBUG] epic_avg={epic_avg_score:.4f}, com_avg={com_avg_score:.4f}, gap={gap_epic_to_com:.4f}, has_disc={should_use_pairwise_discriminator(override_winner, com_tier)}", flush=True)
                                            # If com is close to epic (<0.20), check com vs epic discriminator
                                            if gap_epic_to_com < 0.20 and should_use_pairwise_discriminator(override_winner, com_tier):
                                                # Use best frame where com scored well (not first frame which may have leg/dirt)
                                                com_best_frame = tier_best_frames[com_tier]['frame_idx']
                                                if DEBUG_THIS_SLOT:
                                                    print(f"[DEBUG] Using com best frame {com_best_frame} (score={tier_best_frames[com_tier]['score']:.4f})", flush=True)
                                                if com_best_frame is not None:
                                                    img_color_com = cv2.imread(os.path.join(buffer_dir, all_files[com_best_frame]))
                                                    if img_color_com is not None:
                                                        roi_color_com = img_color_com[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                                        if roi_color_com.shape[:2] == (SIDE_PX, SIDE_PX):
                                                            com_override = apply_pairwise_discriminator(override_winner, com_tier, roi_color_com)
                                                            if DEBUG_THIS_SLOT:
                                                                print(f"[DEBUG] com_override result: {com_override} (override_winner={override_winner})", flush=True)
                                                            if com_override and com_override != override_winner:
                                                                print(f"[Phase3-Epic] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{com_override} (via epic={epic_tier}, gap={gap_to_winner:.3f})", flush=True)
                                                                winner = com_override
                                                                winner_score = com_avg_score
                                                                return winner, round(winner_score, 4), clean_frames_processed, "[D]"
                                        
                                        print(f"[Phase3-Epic] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap_to_winner:.3f})", flush=True)
                                        winner = override_winner
                                        winner_score = epic_avg_score
                                        # Discriminator overrides are high-confidence - bypass threshold check
                                        # Discriminators use strong features (e.g., 108% hue diff for epic1 vs dirt1)
                                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"  # [D] = Discriminator override
                                    break  # Only check first close epic tier
        
        # PHASE 3-DIRT: Check if dirt tier lost to epic/com but should have won
        # Handles cases where fairy overlap causes wrong tier to dominate early frames
        if winner in ['epic1', 'epic2', 'epic3', 'epic4', 'com1', 'com2', 'com3', 'com4']:
            # Check if any dirt tier was a candidate and is close using tier_totals
            for dirt_tier in DIRT_TIERS:
                if dirt_tier in tier_totals and dirt_tier in allowed_tiers:
                    dirt_avg_score = tier_totals[dirt_tier] / tier_frame_counts[dirt_tier]
                    gap_to_winner = winner_score - dirt_avg_score
                    
                    # If dirt tier is close (<0.25) and was in allowed tiers, run discriminator
                    # F55: epic3=0.349 vs dirt3=0.489 avg, but epic3 won first 3 frames
                    # F36: dirt3=0.286 vs com3=0.440 avg (smaller sample), need wider check
                    if gap_to_winner < 0.25:
                        if should_use_pairwise_discriminator(winner, dirt_tier):
                            # Load first clean frame in color for discriminator
                            first_frame_idx = sample_indices[0]
                            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                            if img_color is not None:
                                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                                    override_winner = apply_pairwise_discriminator(winner, dirt_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        print(f"[Phase3-DirtReverse] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap_to_winner:.3f})", flush=True)
                                        winner = override_winner
                                        winner_score = dirt_avg_score
                                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"
                                    break  # Only check first close dirt tier
        
        # PHASE 3-COM: Check if com/leg tier lost to dirt but should have won
        # Handles cases where dirt wins but com/epic/leg actually present (F36: dirt3 wins but com3 correct)
        if winner in DIRT_TIERS:
            # Check ALL close com/epic/leg tiers and find the best discriminator match
            best_override = None
            best_gap = float('inf')
            best_score = None
            
            # Load first clean frame in color once for all discriminators
            first_frame_idx = sample_indices[0]
            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
            
            if img_color is not None:
                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                    # Check all candidate tiers
                    for check_tier in ['com1', 'com2', 'com3', 'com4', 'epic1', 'epic2', 'epic3', 'epic4', 'leg1', 'leg2', 'leg3', 'leg4']:
                        if check_tier in tier_totals and check_tier in allowed_tiers:
                            check_avg_score = tier_totals[check_tier] / tier_frame_counts[check_tier]
                            gap = abs(winner_score - check_avg_score)
                            
                            # If tier is close (<0.15), test discriminator
                            # F36: dirt3=0.286 vs com3=0.375 (gap=0.089)
                            # F54 R1_S0: dirt3 vs leg3/com3, need to check both
                            if gap < 0.15:
                                if should_use_pairwise_discriminator(winner, check_tier):
                                    override_winner = apply_pairwise_discriminator(winner, check_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        # Track the tier with smallest gap (most confident)
                                        if gap < best_gap:
                                            best_override = override_winner
                                            best_gap = gap
                                            best_score = check_avg_score
            
            # Apply best override if found
            if best_override:
                print(f"[Phase3-ComReverse] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{best_override} (gap={best_gap:.3f})", flush=True)
                winner = best_override
                winner_score = best_score
                return winner, round(winner_score, 4), clean_frames_processed, "[D]"
        
        # PHASE 3-RARE: Check if rare1 lost to epic1 but should have won
        # Similar to Phase3-Epic, but for rare1 blocks on F24
        if winner in ['epic1', 'epic2', 'epic3', 'epic4']:
            # Check if rare1 was a candidate and is close
            for rare_tier in ['rare1', 'rare2', 'rare3', 'rare4']:
                if rare_tier in votes:
                    rare_vote_count = votes[rare_tier]
                    rare_avg_score = rare_vote_count / clean_frames_processed
                    gap_to_winner = winner_score - rare_avg_score
                    
                    # If rare tier is close (<0.20) and was in allowed tiers, run discriminator
                    if gap_to_winner < 0.20 and rare_tier in allowed_tiers:
                        if should_use_pairwise_discriminator(winner, rare_tier):
                            # Load first frame in color for discriminator
                            first_frame_idx = sample_indices[0]
                            img_color = cv2.imread(os.path.join(buffer_dir, all_files[first_frame_idx]))
                            if img_color is not None:
                                roi_color = img_color[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                                if roi_color.shape[:2] == (SIDE_PX, SIDE_PX):
                                    override_winner = apply_pairwise_discriminator(winner, rare_tier, roi_color)
                                    if override_winner and override_winner != winner:
                                        print(f"[Phase3-Rare] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{override_winner} (gap={gap_to_winner:.3f})", flush=True)
                                        winner = override_winner
                                        winner_score = rare_avg_score
                                        # Discriminator overrides are high-confidence - bypass threshold check
                                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"
                                    break  # Only check first close rare tier
        
        # PHASE 3-MYTH: Promote myth tiers when they're close to lower-value tiers
        # Myth blocks vary significantly across damage states, causing vote fragmentation
        # When myth is close to com/rare/epic, prefer myth (higher value tier)
        LOWER_TIERS = {'com1', 'com2', 'com3', 'com4', 'rare1', 'rare2', 'rare3', 'rare4', 
                       'epic1', 'epic2', 'epic3', 'epic4'}
        
        if winner not in MYTH_TIERS and winner in LOWER_TIERS:
            for myth_tier in MYTH_TIERS:
                if myth_tier in votes and myth_tier in allowed_tiers:
                    myth_vote_count = votes[myth_tier]
                    myth_avg_score = myth_vote_count / clean_frames_processed
                    gap_to_winner = winner_score - myth_avg_score
                    
                    # If myth is within 0.16 of winner and above minimum quality (0.25), promote it
                    # This catches cases where com2 wins 7 frames at 0.48 each vs myth1 winning 5 frames at 0.50 each
                    # F22 R4 S3: rare1=0.4051 vs myth1=0.2532 (gap=0.152) needs wider threshold
                    if gap_to_winner < 0.16 and myth_avg_score >= 0.25:
                        print(f"[Phase3-Myth] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{myth_tier} (gap={gap_to_winner:.3f}, myth_avg={myth_avg_score:.3f})", flush=True)
                        winner = myth_tier
                        winner_score = myth_avg_score
                        # Return with [M] tag to bypass confidence check (myth1 is valuable even at lower confidence)
                        return winner, round(winner_score, 4), clean_frames_processed, "[M]"
                    break  # Only check first myth tier
        
        # PHASE 3-DIV: Check if div won early due to fairy contamination
        # F60 R1_S0: div1 wins first 5 frames (fairy overlap), then epic/rare appear
        if winner in ['div1', 'div2', 'div3', 'div4'] and clean_frames_processed < 10:
            # Check if epic/rare/leg are in tier_totals (appeared later after div contamination)
            for check_tier in ['epic1', 'epic2', 'epic3', 'epic4', 'rare1', 'rare2', 'rare3', 'rare4', 'leg1', 'leg2', 'leg3', 'leg4']:
                if check_tier in tier_totals and check_tier in allowed_tiers:
                    check_avg_score = tier_totals[check_tier] / tier_frame_counts[check_tier]
                    gap = abs(winner_score - check_avg_score)
                    # If other tier scored close, it's probably the real block
                    if gap < 0.25 and tier_frame_counts[check_tier] >= 3:
                        if DEBUG_THIS_SLOT:
                            print(f"[DEBUG] Phase3-Div: div early contamination, {check_tier} scored {check_avg_score:.4f} over {tier_frame_counts[check_tier]} frames", flush=True)
                        print(f"[Phase3-DivContamination] F{f_id} R{r_idx+1}_S{col_idx}: {winner}→{check_tier} (early fairy contamination)", flush=True)
                        winner = check_tier
                        winner_score = check_avg_score
                        return winner, round(winner_score, 4), clean_frames_processed, "[D]"
        
        # CONFIDENCE CHECK: Now check if final winner meets threshold
        # SPECIAL HANDLING for tiers with template quality issues:
        # - DIRT: Good blocks score 0.39+, damaged 0.05-0.08 → use 0.20 threshold
        # - LEG: Short floors + only wins pristine frames → use 0.20 threshold (same as dirt)
        #   Example: 6-frame floor, leg1 wins all 6 at 0.45 each, but runner-up votes reduce effective average
        # - COM2/COM3: Consistent 0.27-0.35 scores (template quality issue) → use 0.30 threshold
        # - MYTH1/MYTH2: Varies across damage states, vote fragmentation → use 0.20 threshold (F38 R2 S0: myth2=0.224)
        # - RARE1: Marginal scores on some floors (F24: 0.377) → use 0.35 threshold
        # - EPIC2: Fairy/crosshair interference, low scores → use 0.25 threshold (F39: epic2=0.276)
        if winner in DIRT_TIERS:
            MIN_CONFIDENCE = 0.20
        elif winner in LEG_TIERS:
            MIN_CONFIDENCE = 0.20  # Lowered from 0.25 to 0.20 to match dirt handling
        elif winner in ['com1', 'com2', 'com3', 'com4']:
            MIN_CONFIDENCE = 0.26  # Lower than normal due to template quality (F36: com3=0.279, F59: com3=0.263)
        elif winner in MYTH_TIERS:
            MIN_CONFIDENCE = 0.20  # Lower than normal due to damage variation and vote fragmentation
        elif winner in ['rare1', 'rare2', 'rare3', 'rare4']:
            MIN_CONFIDENCE = 0.25  # Lower for fairy/player obstruction and vote fragmentation (F60: rare3=0.252, F62: rare3=0.291)
        elif winner in ['epic1', 'epic2', 'epic3', 'epic4']:
            MIN_CONFIDENCE = 0.15  # Lower for fairy/crosshair interference and vote fragmentation (F39: epic2=0.276, F53: epic3=0.162)
        elif winner in ['div1', 'div2', 'div3', 'div4']:
            MIN_CONFIDENCE = 0.38  # Lower for cases with limited clean frames (F60: div1=0.400 with only 5 frames)
        else:
            MIN_CONFIDENCE = 0.40
        
        # Check if winner meets confidence threshold
        if winner_score < MIN_CONFIDENCE:
            if DEBUG_THIS_SLOT:
                print(f"[DEBUG-FINAL] FAILED confidence check: {winner_score:.4f} < {MIN_CONFIDENCE} → returning low_conf", flush=True)
            return "low_conf", round(winner_score, 4), clean_frames_processed, "[L]"
        
        if DEBUG_THIS_SLOT:
            print(f"[DEBUG-FINAL] PASSED confidence check: {winner_score:.4f} >= {MIN_CONFIDENCE} → returning {winner}", flush=True)
        
        # Final Verification: Divide score by clean frames only
        return winner, round(winner_score, 4), clean_frames_processed, "[M]"

    if DEBUG_THIS_SLOT:
        print(f"[DEBUG-FINAL] No clean frames processed → returning low_conf", flush=True)
    
    return "low_conf", 0, 0, ""

def process_floor_tier(floor_data, dna_map, homing_map, buffer_dir, all_files, res, asc_level=None):
    # Ensure this worker has correct ascension configuration
    if asc_level:
        cfg.set_ascension(asc_level)
    
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    # Debug boss floor check for floor 149
    if f_id == 149:
        import sys
        print(f"DEBUG F149: hasattr(cfg, 'BOSS_DATA')={hasattr(cfg, 'BOSS_DATA')}", file=sys.stderr, flush=True)
        print(f"DEBUG F149: f_id in cfg.BOSS_DATA={f_id in cfg.BOSS_DATA}", file=sys.stderr, flush=True)
        print(f"DEBUG F149: cfg.BOSS_DATA keys={list(cfg.BOSS_DATA.keys()) if hasattr(cfg, 'BOSS_DATA') else 'N/A'}", file=sys.stderr, flush=True)
    
    if hasattr(cfg, 'BOSS_DATA') and f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            results[f"R{r+1}_S{c}"] = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}_tag"] = "[B]"
            # Add score/mom placeholders for consistency
            results[f"R{r+1}_S{c}_score"] = 1.0
            results[f"R{r+1}_S{c}_mom"] = 24
        return results

    allowed =[t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = list(range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1))
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key], results[f"{key}_tag"] = "empty", ""
            else:
                tier, score, mom, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res, homing_map, f_id)
                results[key], results[f"{key}_score"], results[f"{key}_mom"], results[f"{key}_tag"] = tier, score, mom, tag
    return results

def run_tier_identification():
    # Note: Discriminative masks disabled - custom weighted correlation incompatible with OpenCV scoring
    # Note: auto_configure_ascension() is now called at module import time
    print(f"--- STEP 6: TIER CONSENSUS ENGINE (Run {RUN_ID}) ---")
    if not os.path.exists(DNA_INVENTORY_CSV) or not os.path.exists(HOMING_CSV):
        print(f"Error: Missing dependency CSVs for Run {RUN_ID}")
        return

    df_dna, df_homing = pd.read_csv(DNA_INVENTORY_CSV), pd.read_csv(HOMING_CSV)
    homing_map = df_homing.set_index('frame_idx')['slot_id'].to_dict()
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    
    # RANGE BOUNDING: Dynamic application
    if START_FLOOR_BOUND is not None:
        df_floors = df_floors[df_floors['floor_id'] >= START_FLOOR_BOUND]
    if END_FLOOR_BOUND is not None:
        df_floors = df_floors[df_floors['floor_id'] <= END_FLOOR_BOUND]
    
    buffer_dir = SOURCE_DIR
    res = load_all_templates()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    # CRITICAL: Hardcode asc2 for workers instead of passing as parameter
    # Multiprocessing serialization was mangling the string
    worker = partial(process_floor_tier, dna_map=df_dna, homing_map=homing_map, buffer_dir=buffer_dir, all_files=all_files, res=res, asc_level=None)  # Don't pass asc_level, already configured
    inventory =[]
    
    total = len(df_floors)
    print(f"Executing parallel scan on {total} floors...")
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to share memory state
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            result = future.result()
            inventory.append(result)
            print(f"  Processed ({count}/{total}) Floor {result['floor_id']:03d}", end="\r")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    
    # If processing a subset, warn the user. Otherwise, output to the main CSV.
    if START_FLOOR_BOUND is not None or END_FLOOR_BOUND is not None:
        print("\n[!] WARNING: Outputting a partial run. This will overwrite the main inventory CSV.")
        
    final_df.to_csv(OUT_CSV, index=False)
    
    print("\nGenerating Production Proofs...")
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                
                # Production Color Logic
                if tag == "[L]": color = (255, 255, 0)      # Cyan
                elif tag == "[O]": color = (0, 255, 255)    # Yellow
                elif tier == "low_conf": color = (0, 0, 255) # Red
                else: color = (0, 255, 0)                   # Green
                
                # Cleanup: No Momentum/Boss tags in proofs
                clean_label = tier if tag in ["[M]", "[B]"] else f"{tier}{tag}"
                
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, clean_label, (cx-25, cy+HUD_DY), 0, 0.35, (0,0,0), 2)
                cv2.putText(img, clean_label, (cx-25, cy+HUD_DY), 0, 0.35, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"\n[COMPLETE] Master Run finished. Data: {os.path.basename(OUT_CSV)}")

if __name__ == "__main__":
    run_tier_identification()