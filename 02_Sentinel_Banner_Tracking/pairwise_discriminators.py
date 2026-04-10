#!/usr/bin/env python3
"""
Two-Phase Discriminators
Phase 1: Global template matching (narrow to top 2-3 candidates)
Phase 2: Pairwise discriminators (use tier-specific features to break ties)
"""

import cv2
import numpy as np

# Pairwise discriminators for confusable tier pairs
# Each function takes an ROI and returns which tier it is

def discriminate_rare1_vs_com1(roi_bgr):
    """
    Discriminate rare1 from com1 using HUE in specific regions (48x48 coordinates).
    
    Based on interactive analysis with resized templates:
    - ROI4 (29,23)→(37,31): Hue diff=50.9% (rare1=97.09, com1=57.72) - STRONGEST
    - ROI1 (32,32)→(42,42): Hue diff=43.6% (rare1=70.69, com1=110.06) - "green ball"
    - ROI2 (14,19)→(23,27): Hue diff=36.8% (rare1=69.64, com1=101.01)
    
    Returns: 'rare1' or 'com1'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_rare1 = 0
    votes_com1 = 0
    
    # Region 4: Right-middle (STRONGEST - 50.9% diff)
    # rare1 has HIGHER hue here (rare1=97.09, com1=57.72)
    region4 = roi_bgr[23:31, 29:37]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 > 77.41:
        votes_rare1 += 2  # Strongest signal gets 2 votes
    else:
        votes_com1 += 2
    
    # Region 1: Lower-right "green ball" (43.6% diff)
    # rare1 has LOWER hue here (rare1=70.69, com1=110.06)
    region1 = roi_bgr[32:42, 32:42]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hue1 = hsv1[:, :, 0].mean()
    
    if hue1 < 90.38:
        votes_rare1 += 1
    else:
        votes_com1 += 1
    
    # Region 2: Upper-left (36.8% diff)
    # rare1 has LOWER hue here (rare1=69.64, com1=101.01)
    region2 = roi_bgr[19:27, 14:23]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    hue2 = hsv2[:, :, 0].mean()
    
    if hue2 < 85.33:
        votes_rare1 += 1
    else:
        votes_com1 += 1
    
    # Decision: majority vote (total 4 votes: 2+1+1)
    if votes_rare1 > votes_com1:
        return 'rare1'
    else:
        return 'com1'

def discriminate_rare1_vs_com2(roi_bgr):
    """
    Discriminate rare1 from com2 using HUE in specific regions (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI1 (14,19)→(22,28): Hue diff=47.9% (rare1=64.54, com2=105.19)
    - ROI2 (31,31)→(42,41): Hue diff=42.8% (rare1=72.35, com2=111.72)
    - ROI4 (30,22)→(38,32): Hue diff=39.1% (rare1=101.41, com2=68.22)
    
    rare1 generally has lower hue in most regions, higher in middle-right.
    
    Returns: 'rare1' or 'com2'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_rare1 = 0
    votes_com2 = 0
    
    # Region 1: Upper-left (47.9% diff)
    # rare1 has LOWER hue (rare1=64.54, com2=105.19)
    region1 = roi_bgr[19:28, 14:22]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hue1 = hsv1[:, :, 0].mean()
    
    if hue1 < 84.87:
        votes_rare1 += 1
    else:
        votes_com2 += 1
    
    # Region 2: Lower-right "green ball" (42.8% diff)
    # rare1 has LOWER hue (rare1=72.35, com2=111.72)
    region2 = roi_bgr[31:41, 31:42]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    hue2 = hsv2[:, :, 0].mean()
    
    if hue2 < 92.03:
        votes_rare1 += 1
    else:
        votes_com2 += 1
    
    # Region 4: Middle-right (39.1% diff)
    # rare1 has HIGHER hue here (rare1=101.41, com2=68.22)
    region4 = roi_bgr[22:32, 30:38]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 > 84.82:
        votes_rare1 += 1
    else:
        votes_com2 += 1
    
    # Decision: majority vote (3 votes total)
    if votes_rare1 >= 2:
        return 'rare1'
    else:
        return 'com2'

def discriminate_epic1_vs_dirt1(roi_bgr):
    """
    Discriminate epic1 from dirt1 using HUE (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI4 (16,26)→(23,34): Hue diff=130.1% (epic1=105.14, dirt1=22.25) - STRONGEST
    
    epic1 has cyan/blue hue, dirt1 has brown/red hue.
    
    Returns: 'epic1' or 'dirt1'
    """
    h, w = roi_bgr.shape[:2]
    
    # Region 4: Center-left (MASSIVE 130% hue difference!)
    region4 = roi_bgr[26:34, 16:23]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    # Threshold: 63.70 (epic1=105.14, dirt1=22.25)
    # epic1 has HIGH hue (cyan), dirt1 has LOW hue (brown)
    if hue4 > 63.70:
        return 'epic1'
    else:
        return 'dirt1'

def discriminate_epic1_vs_com1(roi_bgr):
    """
    Discriminate epic1 from com1.
    
    Based on calibration (20 samples each):
    - mean_green: epic1=91.47, com1=82.29 → epic1 has MORE green (threshold: 86.88)
    - brightness: epic1=88.81, com1=85.04 → epic1 is BRIGHTER (threshold: 86.92)
    - area_ratio: epic1=0.93, com1=0.90 → epic1 is BULKIER (threshold: 0.92)
    
    All features are large effect (***), so epic1 is very distinctive from com1.
    
    Returns: 'epic1' or 'com1'
    """
    h, w = roi_bgr.shape[:2]
    center_region = roi_bgr[int(h*0.4):int(h*0.8), int(w*0.3):int(w*0.7)]
    
    # Feature 1: Green channel
    mean_green = center_region[:, :, 1].mean()
    
    # Feature 2: Brightness
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    mean_brightness = hsv[:, :, 2].mean()
    
    # Feature 3: Area
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, block_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    area_ratio = np.count_nonzero(block_mask) / block_mask.size
    
    # Calibrated thresholds
    green_threshold = 86.88
    brightness_threshold = 86.92
    area_threshold = 0.92
    
    # Use multiple features (if 2+ indicate epic1, call it epic1)
    votes_epic = 0
    if mean_green > green_threshold:
        votes_epic += 1
    if mean_brightness > brightness_threshold:
        votes_epic += 1
    if area_ratio > area_threshold:
        votes_epic += 1
    
    if votes_epic >= 2:
        return 'epic1'
    else:
        return 'com1'

def discriminate_epic1_vs_rare1(roi_bgr):
    """
    Discriminate epic1 from rare1 using SATURATION/HUE/BLUE in multiple regions (48x48).
    
    Analysis shows massive differences:
    - ROI1 (13,19)-(23,29): Saturation 57% diff (rare1=53.96, epic1=97.62)
    - ROI2 (33,30)-(43,42): Blue 59% diff (rare1=68.22, epic1=126.00)
    - ROI3 (13,14)-(22,27): Saturation 49% diff (rare1=63.09, epic1=104.79)
    - ROI5 (24,14)-(32,26): Green 39% diff (rare1=146.44, epic1=98.45)
    
    Returns: 'epic1' or 'rare1'
    """
    votes_rare1, votes_epic1 = 0, 0
    
    # Region 1: Left-center (57% saturation diff - STRONGEST for epic1)
    region1 = roi_bgr[19:29, 13:23]
    sat1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    if sat1 < 75.79:  # rare1=53.96, epic1=97.62
        votes_rare1 += 2  # Low saturation = rare1
    else:
        votes_epic1 += 2  # High saturation = epic1
    
    # Region 2: Right-lower (59% blue diff)
    region2 = roi_bgr[30:42, 33:43]
    blue2 = region2[:, :, 0].mean()
    if blue2 > 97.11:  # rare1=68.22, epic1=126.00
        votes_epic1 += 2  # High blue = epic1
    else:
        votes_rare1 += 2
    
    # Region 5: Center-upper (39% green diff - STRONGEST for rare1)
    region5 = roi_bgr[14:26, 24:32]
    green5 = region5[:, :, 1].mean()
    if green5 > 122.44:  # rare1=146.44, epic1=98.45
        votes_rare1 += 2  # High green = rare1
    else:
        votes_epic1 += 2
    
    return 'epic1' if votes_epic1 > votes_rare1 else 'rare1'

def discriminate_dirt2_vs_com1(roi_bgr):
    """
    Discriminate dirt2 from com1.
    
    dirt2 has brown/orange hue, com1 is gray.
    
    Returns: 'dirt2' or 'com1'
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    h_dim, w_dim = roi_bgr.shape[:2]
    center_region = slice(int(h_dim*0.4), int(h_dim*0.8)), slice(int(w_dim*0.3), int(w_dim*0.7))
    
    # Check for brown/orange hue (hue values around 10-30 in OpenCV)
    hue_vals = h[center_region]
    mean_hue = hue_vals.mean()
    mean_saturation = s[center_region].mean()
    
    # dirt2 has warm hue (orange/brown) and higher saturation
    if mean_saturation > 38 or (10 < mean_hue < 35):
        return 'dirt2'
    else:
        return 'com1'

def discriminate_epic1_vs_com1_hue(roi_bgr):
    """
    Discriminate epic1 from com1 using SATURATION/HUE (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI1 (13,14)→(21,25): Saturation diff=46.7% (epic1=116.17, com1=72.22)
    - ROI2 (30,27)→(41,37): Saturation diff=40.2% (epic1=99.29, com1=66.05)
    - ROI4 (27,22)→(38,31): Hue diff=37.9% (epic1=110.61, com1=75.36)
    
    epic1 is much more saturated/vibrant than com1.
    
    Returns: 'epic1' or 'com1'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_epic1 = 0
    votes_com1 = 0
    
    # Region 1: Upper area (46.7% saturation diff)
    region1 = roi_bgr[14:25, 13:21]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    sat1 = hsv1[:, :, 1].mean()
    
    if sat1 > 94.19:
        votes_epic1 += 2  # Strongest signal
    else:
        votes_com1 += 2
    
    # Region 2: Lower-right (40.2% saturation diff)
    region2 = roi_bgr[27:37, 30:41]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    sat2 = hsv2[:, :, 1].mean()
    
    if sat2 > 82.67:
        votes_epic1 += 1
    else:
        votes_com1 += 1
    
    # Region 4: Middle area (37.9% hue diff)
    region4 = roi_bgr[22:31, 27:38]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 > 92.98:
        votes_epic1 += 1
    else:
        votes_com1 += 1
    
    # Decision: majority vote (4 votes total: 2+1+1)
    if votes_epic1 > votes_com1:
        return 'epic1'
    else:
        return 'com1'

def discriminate_dirt2_vs_com1_brightness(roi_bgr):
    """
    Discriminate dirt2 from com1 using BRIGHTNESS (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI1 (15,22)→(24,30): Green=60%, Blue=54%, Brightness=59% (dirt2 MUCH BRIGHTER)
    - ROI4 (29,24)→(38,32): Hue diff=45.6% (dirt2=108.43, com1=68.14)
    
    dirt2 is whitish/chalky, com1 is darker gray.
    
    Returns: 'dirt2' or 'com1'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_dirt2 = 0
    votes_com1 = 0
    
    # Region 1: Center (MASSIVE brightness diff)
    region1 = roi_bgr[22:30, 15:24]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    brightness1 = hsv1[:, :, 2].mean()
    
    if brightness1 > 116.29:
        votes_dirt2 += 2  # Strongest signal
    else:
        votes_com1 += 2
    
    # Region 4: Right area (45.6% hue diff)
    region4 = roi_bgr[24:32, 29:38]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 > 88.28:
        votes_dirt2 += 1
    else:
        votes_com1 += 1
    
    # Decision: majority vote (3 votes total)
    if votes_dirt2 >= 2:
        return 'dirt2'
    else:
        return 'com1'

def discriminate_div1_vs_rare3(roi_bgr):
    """
    Discriminate div1 from rare3 using BRIGHTNESS (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI2 (26,20)→(35,30): Value diff=69.3% (div1=167.24, rare3=81.14)
    - ROI3 (22,32)→(30,40): Value diff=83.7% (div1=152.95, rare3=62.69)
    
    div1 blocks are GLOWING/LUMINOUS, rare3 are normal.
    
    Returns: 'div1' or 'rare3'
    """
    h, w = roi_bgr.shape[:2]
    
    # Region 2: Upper area (69% brightness diff)
    region2 = roi_bgr[20:30, 26:35]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    value2 = hsv2[:, :, 2].mean()
    
    # Threshold: 124.19 (div1=167, rare3=81)
    # div blocks are MUCH brighter
    if value2 > 124.19:
        return 'div1'
    else:
        return 'rare3'

def discriminate_div2_vs_rare3(roi_bgr):
    """
    Discriminate div2 from rare3 using BRIGHTNESS (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI2 (24,19)→(34,31): Value diff=83.8% (div2=200.64, rare3=82.12)
    
    div2 blocks are GLOWING/LUMINOUS, rare3 are normal.
    
    Returns: 'div2' or 'rare3'
    """
    h, w = roi_bgr.shape[:2]
    
    # Region 2: Main area (MASSIVE 83.8% brightness diff)
    region2 = roi_bgr[19:31, 24:34]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    value2 = hsv2[:, :, 2].mean()
    
    # Threshold: 141.38 (div2=200, rare3=82)
    if value2 > 141.38:
        return 'div2'
    else:
        return 'rare3'

def discriminate_div3_vs_leg3(roi_bgr):
    """
    Discriminate div3 from leg3 using BRIGHTNESS (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI3 (21,23)→(35,34): Value diff=69.1% (div3=198.47, leg3=96.55)
    - ROI4 (25,30)→(44,42): Value diff=47.9% (div3=165.85, leg3=101.73)
    
    div3 blocks are GLOWING/LUMINOUS, leg3 are darker.
    
    Returns: 'div3' or 'leg3'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_div3 = 0
    votes_leg3 = 0
    
    # Region 3: Center (69% brightness diff)
    region3 = roi_bgr[23:34, 21:35]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    value3 = hsv3[:, :, 2].mean()
    
    if value3 > 147.51:
        votes_div3 += 2
    else:
        votes_leg3 += 2
    
    # Region 4: Lower area (48% brightness diff)
    region4 = roi_bgr[30:42, 25:44]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    value4 = hsv4[:, :, 2].mean()
    
    if value4 > 133.79:
        votes_div3 += 1
    else:
        votes_leg3 += 1
    
    # Decision: majority vote (3 votes total)
    if votes_div3 >= 2:
        return 'div3'
    else:
        return 'leg3'

def discriminate_div3_vs_epic4(roi_bgr):
    """
    Discriminate div3 from epic4 using BRIGHTNESS (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI3 (22,22)→(31,35): Value diff=72.3% (div3=191.15, epic4=89.62)
    - ROI2 (6,20)→(17,30): Value diff=30.9% (div3=176.90, epic4=129.50)
    
    div3 blocks are GLOWING/LUMINOUS, epic4 are darker.
    
    Returns: 'div3' or 'epic4'
    """
    h, w = roi_bgr.shape[:2]
    
    # Region 3: Center (MASSIVE 72% brightness diff)
    region3 = roi_bgr[22:35, 22:31]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    value3 = hsv3[:, :, 2].mean()
    
    # Threshold: 140.38 (div3=191, epic4=90)
    if value3 > 140.38:
        return 'div3'
    else:
        return 'epic4'

def discriminate_epic1_vs_dirt2(roi_bgr):
    """
    Discriminate epic1 from dirt2 (48x48 coordinates).
    
    NEEDS INTERACTIVE ANALYSIS - placeholder using known features:
    - epic1: Purple/magenta gem, high saturation
    - dirt2: Whitish/chalky brown, high brightness but low saturation
    
    Returns: 'epic1' or 'dirt2'
    """
    h, w = roi_bgr.shape[:2]
    
    # Use center region
    center_region = roi_bgr[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    
    # epic1 has HIGH saturation (vibrant purple), dirt2 has LOW saturation (whitish)
    saturation = hsv[:, :, 1].mean()
    brightness = hsv[:, :, 2].mean()
    
    # epic1: high sat (90+), moderate brightness (80-100)
    # dirt2: low sat (40-60), high brightness (110+)
    
    if saturation > 70:
        return 'epic1'  # High saturation = epic1
    elif brightness > 105 and saturation < 60:
        return 'dirt2'  # Bright + low sat = dirt2
    else:
        # Default to saturation (most discriminative)
        return 'epic1' if saturation > 55 else 'dirt2'

def discriminate_dirt2_vs_com2(roi_bgr):
    """
    Discriminate dirt2 from com2 using BRIGHTNESS/HUE (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI1 (16,22)→(24,31): Brightness=87.2% diff (dirt2=151.74, com2=59.57)
    - ROI2 (6,32)→(14,41): Green=51.7% diff (dirt2=102.58, com2=60.42)
    - ROI4 (31,21)→(39,32): Hue=37.9% diff (dirt2=107.34, com2=73.10)
    
    dirt2 is MUCH brighter/whiter than com2.
    
    Returns: 'dirt2' or 'com2'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_dirt2 = 0
    votes_com2 = 0
    
    # Region 1: Upper-left (MASSIVE 87% brightness diff)
    region1 = roi_bgr[22:31, 16:24]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    brightness1 = hsv1[:, :, 2].mean()
    
    if brightness1 > 105.65:
        votes_dirt2 += 2  # Strongest signal
    else:
        votes_com2 += 2
    
    # Region 2: Lower area (51.7% green diff)
    region2 = roi_bgr[32:41, 6:14]
    green2 = region2[:, :, 1].mean()
    
    if green2 > 81.50:
        votes_dirt2 += 1
    else:
        votes_com2 += 1
    
    # Region 4: Right side (37.9% hue diff)
    region4 = roi_bgr[21:32, 31:39]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 > 90.22:
        votes_dirt2 += 1
    else:
        votes_com2 += 1
    
    # Decision: majority vote (4 votes total: 2+1+1)
    if votes_dirt2 >= 3:
        return 'dirt2'
    else:
        return 'com2'

def discriminate_epic1_vs_dirt1(roi_bgr):
    """
    Discriminate epic1 from dirt1 using HUE/BLUE in multiple regions (48x48 coordinates).
    
    Analysis shows massive differences:
    - ROI4 (15,28)-(21,34): Hue 108% diff, Blue 70% diff  
    - ROI5 (25,27)-(33,34): Blue 83% diff, Hue 36% diff
    - ROI3 (14,31)-(22,37): Green 58% diff, Blue 66% diff
    """
    votes_epic1, votes_dirt1 = 0, 0
    
    # Region 4: Center-left (STRONGEST - 108% hue diff)
    region4 = roi_bgr[28:34, 15:21]
    hue4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)[:, :, 0].mean()
    if hue4 > 70.82:  # epic1=109.31, dirt1=32.33
        votes_epic1 += 2  # Strongest signal
    else:
        votes_dirt1 += 2
    
    # Region 5: Center-right (83% blue diff)
    region5 = roi_bgr[27:34, 25:33]
    blue5 = region5[:, :, 0].mean()
    if blue5 > 50.61:  # epic1=71.73, dirt1=29.48
        votes_epic1 += 2
    else:
        votes_dirt1 += 2
    
    # Region 3: Lower-left (66% blue diff)
    region3 = roi_bgr[31:37, 14:22]
    green3 = region3[:, :, 1].mean()
    if green3 > 65.42:  # epic1=84.44, dirt1=46.40
        votes_epic1 += 1
    else:
        votes_dirt1 += 1
    
    return 'epic1' if votes_epic1 > votes_dirt1 else 'dirt1'


def discriminate_leg1_vs_epic1(roi_bgr):
    """
    Discriminate leg1 from epic1 using HUE/SATURATION (48x48 coordinates).
    
    Based on interactive analysis:
    - ROI1 (28,31)→(39,40): Hue=76.2% diff (leg1=52.24, epic1=116.51)
    - ROI4 (28,26)→(40,38): Hue=42.6% diff (leg1=75.57, epic1=116.44)
    - ROI3 (12,14)→(22,26): Saturation=34.3% diff (leg1=74.17, epic1=104.92)
    
    epic1 has HIGH hue (cyan/blue), leg1 has LOW hue (orange/red).
    
    Returns: 'leg1' or 'epic1'
    """
    h, w = roi_bgr.shape[:2]
    
    votes_leg1 = 0
    votes_epic1 = 0
    
    # Region 1: Lower-right (MASSIVE 76% hue diff!)
    region1 = roi_bgr[31:40, 28:39]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hue1 = hsv1[:, :, 0].mean()
    
    if hue1 < 84.37:
        votes_leg1 += 2  # Strongest signal
    else:
        votes_epic1 += 2
    
    # Region 4: Middle-right (42.6% hue diff)
    region4 = roi_bgr[26:38, 28:40]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    
    if hue4 < 96.01:
        votes_leg1 += 1
    else:
        votes_epic1 += 1
    
    # Region 3: Upper area (34.3% saturation diff)
    region3 = roi_bgr[14:26, 12:22]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    sat3 = hsv3[:, :, 1].mean()
    
    if sat3 < 89.55:
        votes_leg1 += 1
    else:
        votes_epic1 += 1
    
    # Decision: majority vote (4 votes total: 2+1+1)
    if votes_leg1 >= 3:
        return 'leg1'
    else:
        return 'epic1'

# Registry of all pairwise discriminators
PAIRWISE_DISCRIMINATORS = {
    ('rare1', 'com1'): discriminate_rare1_vs_com1,
    ('com1', 'rare1'): discriminate_rare1_vs_com1,
    ('rare1', 'com2'): discriminate_rare1_vs_com2,
    ('com2', 'rare1'): discriminate_rare1_vs_com2,
    ('rare1', 'epic1'): discriminate_epic1_vs_rare1,
    ('epic1', 'rare1'): discriminate_epic1_vs_rare1,
    ('epic1', 'dirt1'): discriminate_epic1_vs_dirt1,
    ('dirt1', 'epic1'): discriminate_epic1_vs_dirt1,
    ('epic1', 'dirt2'): discriminate_epic1_vs_dirt2,
    ('dirt2', 'epic1'): discriminate_epic1_vs_dirt2,
    ('epic1', 'com1'): discriminate_epic1_vs_com1_hue,
    ('com1', 'epic1'): discriminate_epic1_vs_com1_hue,
    ('dirt2', 'com1'): discriminate_dirt2_vs_com1_brightness,
    ('com1', 'dirt2'): discriminate_dirt2_vs_com1_brightness,
    ('dirt2', 'com2'): discriminate_dirt2_vs_com2,
    ('com2', 'dirt2'): discriminate_dirt2_vs_com2,
    ('leg1', 'epic1'): discriminate_leg1_vs_epic1,
    ('epic1', 'leg1'): discriminate_leg1_vs_epic1,
    ('div1', 'rare3'): discriminate_div1_vs_rare3,
    ('rare3', 'div1'): discriminate_div1_vs_rare3,
    ('div2', 'rare3'): discriminate_div2_vs_rare3,
    ('rare3', 'div2'): discriminate_div2_vs_rare3,
    ('div3', 'leg3'): discriminate_div3_vs_leg3,
    ('leg3', 'div3'): discriminate_div3_vs_leg3,
    ('div3', 'epic4'): discriminate_div3_vs_epic4,
    ('epic4', 'div3'): discriminate_div3_vs_epic4,
}

def apply_pairwise_discriminator(tier1, tier2, roi_bgr):
    """
    Apply pairwise discriminator if one exists.
    
    Args:
        tier1, tier2: The two candidate tiers
        roi_bgr: ROI in BGR format
    
    Returns:
        Winner tier, or None if no discriminator exists
    """
    key = (tier1, tier2)
    if key in PAIRWISE_DISCRIMINATORS:
        discriminator = PAIRWISE_DISCRIMINATORS[key]
        winner = discriminator(roi_bgr)
        return winner
    return None

def should_use_pairwise_discriminator(tier1, tier2):
    """
    Determine if pairwise discriminator exists for this pair.
    
    Note: Gap checking is done by caller, this just checks if discriminator exists.
    
    Returns:
        Boolean
    """
    return (tier1, tier2) in PAIRWISE_DISCRIMINATORS

if __name__ == '__main__':
    print("""
================================================================================
TWO-PHASE DISCRIMINATION SYSTEM
================================================================================

Phase 1: Global Template Matching
  - Use standard OpenCV TM_CCOEFF_NORMED
  - Get top 2-3 candidates with scores
  
Phase 2: Pairwise Discriminators (if scores are close)
  - If gap < 0.05, apply tier-specific discriminator
  - Use visual features like color, saturation, specific pixel regions
  
Available Pairwise Discriminators:
================================================================================
""")
    
    for (tier1, tier2), func in sorted(PAIRWISE_DISCRIMINATORS.items()):
        # Only print each pair once (not both orderings)
        if tier1 < tier2:
            print(f"  {tier1:8s} vs {tier2:8s} : {func.__name__}")
    
    print(f"""
Total: {len(PAIRWISE_DISCRIMINATORS) // 2} discriminators

How It Works:
-------------
1. Step6 does global template matching → top candidates with scores
2. If top 2 candidates have gap < 0.05:
   - Check if pairwise discriminator exists
   - If yes, use discriminator to break tie
   - Discriminator uses tier-specific features (green ball, saturation, etc.)
3. Winner is returned

Integration:
------------
In step6_tier_consensus.py identify_consensus():
  
  # After voting determines top 2 candidates
  winner, runner_up = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:2]
  gap = winner[1] - runner_up[1]
  
  if gap < 0.05:
      # Try pairwise discriminator
      roi_bgr = cv2.imread(...)  # Load in color
      override = apply_pairwise_discriminator(winner[0], runner_up[0], roi_bgr)
      if override:
          winner = (override, winner[1])
""")
