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

def discriminate_com3_vs_dirt3(roi_bgr):
    """
    Discriminate between com3 and dirt3 blocks.
    
    Key pattern: com3 has lower hue values (orange/yellow tones),
    dirt3 has higher hue values (green/cyan tones).
    
    Uses voting across multiple regions for robustness.
    """
    votes_com3, votes_dirt3 = 0, 0
    
    # Region 1: Upper-left (24,14)-(33,24) - 51.5% hue diff
    region1 = roi_bgr[14:24, 24:33]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hue1 = hsv1[:, :, 0].mean()
    if hue1 < 91.18:  # Midpoint between com3=67.69 and dirt3=114.68
        votes_com3 += 2  # Strong signal
    else:
        votes_dirt3 += 2
    
    # Region 2: Lower-right (38,34)-(46,41) - 60.4% hue diff, 37.1% blue diff
    region2 = roi_bgr[34:41, 38:46]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    hue2 = hsv2[:, :, 0].mean()
    blue2 = region2[:, :, 0].mean()
    if hue2 < 79.03:  # Midpoint
        votes_com3 += 2
    else:
        votes_dirt3 += 2
    if blue2 < 100.90:  # Midpoint between com3=82.18 and dirt3=119.62
        votes_com3 += 1
    else:
        votes_dirt3 += 1
    
    # Region 3: Lower-left (21,34)-(28,42) - 66.3% hue diff
    region3 = roi_bgr[34:42, 21:28]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    hue3 = hsv3[:, :, 0].mean()
    if hue3 < 74.49:  # Midpoint
        votes_com3 += 2
    else:
        votes_dirt3 += 2
    
    # Region 5: Center (19,20)-(29,27) - 26.4% hue diff, 24.2% blue diff
    region5 = roi_bgr[20:27, 19:29]
    hsv5 = cv2.cvtColor(region5, cv2.COLOR_BGR2HSV)
    hue5 = hsv5[:, :, 0].mean()
    blue5 = region5[:, :, 0].mean()
    if hue5 < 105.11:  # Midpoint
        votes_com3 += 1
    else:
        votes_dirt3 += 1
    if blue5 < 93.99:  # Midpoint between com3=82.61 and dirt3=105.37
        votes_com3 += 1
    else:
        votes_dirt3 += 1
    
    return 'com3' if votes_com3 > votes_dirt3 else 'dirt3'

def discriminate_dirt3_vs_epic3(roi_bgr):
    """
    Discriminate between dirt3 and epic3 blocks.
    
    Key pattern: epic3 has much higher blue channel values than dirt3.
    Also differs in red channel (epic3 higher) and green channel (dirt3 can be higher).
    
    Uses voting across multiple regions for robustness.
    """
    votes_dirt3, votes_epic3 = 0, 0
    
    # Region 2: Lower section (27,28)-(40,40) - 66.0% blue diff
    region2 = roi_bgr[28:40, 27:40]
    blue2 = region2[:, :, 0].mean()
    red2 = region2[:, :, 2].mean()
    if blue2 < 87.64:  # Midpoint between dirt3=58.70 and epic3=116.58
        votes_dirt3 += 3  # Very strong signal
    else:
        votes_epic3 += 3
    if red2 < 70.56:  # Midpoint between dirt3=55.71 and epic3=85.42
        votes_dirt3 += 1
    else:
        votes_epic3 += 1
    
    # Region 4: Upper-right (28,10)-(37,23) - 35.4% red diff, 29.9% hue diff
    region4 = roi_bgr[10:23, 28:37]
    red4 = region4[:, :, 2].mean()
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    if red4 < 111.20:  # Midpoint between dirt3=91.50 and epic3=130.91
        votes_dirt3 += 2
    else:
        votes_epic3 += 2
    if hue4 > 98.46:  # Midpoint (inverted because dirt3 higher)
        votes_dirt3 += 1
    else:
        votes_epic3 += 1
    
    # Region 5: Lower-right (30,24)-(40,39) - 73.8% blue diff
    region5 = roi_bgr[24:39, 30:40]
    blue5 = region5[:, :, 0].mean()
    red5 = region5[:, :, 2].mean()
    if blue5 < 92.47:  # Midpoint between dirt3=58.33 and epic3=126.61
        votes_dirt3 += 3  # Very strong signal
    else:
        votes_epic3 += 3
    if red5 < 75.60:  # Midpoint between dirt3=59.65 and epic3=91.55
        votes_dirt3 += 1
    else:
        votes_epic3 += 1
    
    return 'dirt3' if votes_dirt3 > votes_epic3 else 'epic3'

def discriminate_leg3_vs_com3(roi_bgr):
    """
    Discriminate between leg3 and com3 blocks.
    
    Key pattern: leg3 has lower hue (orange/red tones), higher red channel values,
    and higher brightness in certain regions. com3 has higher hue (green/cyan tones).
    
    Uses voting across multiple regions for robustness.
    """
    votes_leg3, votes_com3 = 0, 0
    
    # Region 1: Lower section (26,31)-(43,41) - 96.1% hue diff, 61.1% saturation diff
    region1 = roi_bgr[31:41, 26:43]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hue1 = hsv1[:, :, 0].mean()
    sat1 = hsv1[:, :, 1].mean()
    if hue1 < 65.61:  # Midpoint between leg3=34.09 and com3=97.14
        votes_leg3 += 3  # Very strong signal
    else:
        votes_com3 += 3
    if sat1 > 92.26:  # Midpoint between leg3=120.45 and com3=64.06
        votes_leg3 += 2
    else:
        votes_com3 += 2
    
    # Region 4: Lower-left (3,37)-(16,45) - 51.8% hue diff, 30.2% red diff
    region4 = roi_bgr[37:45, 3:16]
    hsv4 = cv2.cvtColor(region4, cv2.COLOR_BGR2HSV)
    hue4 = hsv4[:, :, 0].mean()
    red4 = region4[:, :, 2].mean()
    if hue4 < 85.50:  # Midpoint between leg3=63.35 and com3=107.65
        votes_leg3 += 2
    else:
        votes_com3 += 2
    if red4 > 92.02:  # Midpoint between leg3=105.92 and com3=78.11
        votes_leg3 += 2
    else:
        votes_com3 += 2
    
    # Region 5: Upper section (25,14)-(34,24) - 28.3% value diff, 30.0% green diff
    region5 = roi_bgr[14:24, 25:34]
    hsv5 = cv2.cvtColor(region5, cv2.COLOR_BGR2HSV)
    value5 = hsv5[:, :, 2].mean()
    green5 = region5[:, :, 1].mean()
    if value5 > 135.84:  # Midpoint between leg3=155.04 and com3=116.64
        votes_leg3 += 2
    else:
        votes_com3 += 2
    if green5 > 122.02:  # Midpoint between leg3=140.32 and com3=103.72
        votes_leg3 += 1
    else:
        votes_com3 += 1
    
    # Region 8: Left-lower (4,33)-(16,42) - 32.0% hue diff, 44.6% red diff
    region8 = roi_bgr[33:42, 4:16]
    hsv8 = cv2.cvtColor(region8, cv2.COLOR_BGR2HSV)
    hue8 = hsv8[:, :, 0].mean()
    red8 = region8[:, :, 2].mean()
    if hue8 < 95.82:  # Midpoint between leg3=80.46 and com3=111.17
        votes_leg3 += 2
    else:
        votes_com3 += 2
    if red8 > 75.65:  # Midpoint between leg3=92.51 and com3=58.79
        votes_leg3 += 2
    else:
        votes_com3 += 2
    
    return 'leg3' if votes_leg3 > votes_com3 else 'com3'

def discriminate_dirt3_vs_leg3(roi_bgr):
    """
    Discriminate between dirt3 and leg3 blocks.
    
    Key pattern: leg3 has lower hue (orange/red tones), higher red/green channel values.
    dirt3 has higher hue (green/cyan tones).
    
    Uses voting across multiple regions for robustness.
    """
    votes_dirt3, votes_leg3 = 0, 0
    
    # Region 2: Upper section (26,15)-(33,26) - 58.1% hue diff, 52.2% green diff
    region2 = roi_bgr[15:26, 26:33]
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
    hue2 = hsv2[:, :, 0].mean()
    green2 = region2[:, :, 1].mean()
    if hue2 > 91.58:  # Midpoint between leg3=64.97 and dirt3=118.18
        votes_dirt3 += 3  # Very strong signal
    else:
        votes_leg3 += 3
    if green2 < 105.73:  # Midpoint between leg3=133.30 and dirt3=78.16
        votes_dirt3 += 2
    else:
        votes_leg3 += 2
    
    # Region 3: Lower section (27,31)-(37,40) - 118.5% hue diff (massive!), 76.5% red diff
    region3 = roi_bgr[31:40, 27:37]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    hue3 = hsv3[:, :, 0].mean()
    red3 = region3[:, :, 2].mean()
    if hue3 > 70.78:  # Midpoint between leg3=28.84 and dirt3=112.71
        votes_dirt3 += 3  # Extremely strong signal
    else:
        votes_leg3 += 3
    if red3 < 90.22:  # Midpoint between leg3=124.71 and dirt3=55.72
        votes_dirt3 += 2
    else:
        votes_leg3 += 2
    
    # Region 6: Center (24,18)-(36,28) - 61.2% hue diff, 52.1% red diff
    region6 = roi_bgr[18:28, 24:36]
    hsv6 = cv2.cvtColor(region6, cv2.COLOR_BGR2HSV)
    hue6 = hsv6[:, :, 0].mean()
    red6 = region6[:, :, 2].mean()
    if hue6 > 92.79:  # Midpoint between leg3=64.38 and dirt3=121.20
        votes_dirt3 += 2
    else:
        votes_leg3 += 2
    if red6 < 89.62:  # Midpoint between leg3=112.96 and dirt3=66.28
        votes_dirt3 += 2
    else:
        votes_leg3 += 2
    
    # Region 7: Lower-wide (25,32)-(44,41) - 99.4% hue diff, 42.1% red diff
    region7 = roi_bgr[32:41, 25:44]
    hsv7 = cv2.cvtColor(region7, cv2.COLOR_BGR2HSV)
    hue7 = hsv7[:, :, 0].mean()
    red7 = region7[:, :, 2].mean()
    if hue7 > 72.36:  # Midpoint between leg3=36.40 and dirt3=108.32
        votes_dirt3 += 3  # Very strong signal
    else:
        votes_leg3 += 3
    if red7 < 89.87:  # Midpoint between leg3=108.78 and dirt3=70.95
        votes_dirt3 += 1
    else:
        votes_leg3 += 1
    
    return 'dirt3' if votes_dirt3 > votes_leg3 else 'leg3'

def discriminate_com3_vs_epic3(roi_bgr):
    """
    Discriminate between com3 and epic3 blocks.
    
    Key pattern: epic3 has much higher saturation (90-120) and blue channel values.
    com3 has lower saturation (55-75) and higher green values in some regions.
    
    Uses voting across multiple regions for robustness.
    """
    votes_com3, votes_epic3 = 0, 0
    
    # Region 1: Left-center (6,21)-(15,30) - 58.9% saturation diff, 42.2% hue diff
    region1 = roi_bgr[21:30, 6:15]
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    sat1 = hsv1[:, :, 1].mean()
    hue1 = hsv1[:, :, 0].mean()
    if sat1 < 94.08:  # Midpoint between com3=66.38 and epic3=121.78
        votes_com3 += 3  # Very strong signal
    else:
        votes_epic3 += 3
    if hue1 < 107.00:  # Midpoint between com3=84.44 and epic3=129.54
        votes_com3 += 2
    else:
        votes_epic3 += 2
    
    # Region 3: Lower-right (37,32)-(45,42) - 53.6% saturation diff, 39.1% hue diff
    region3 = roi_bgr[32:42, 37:45]
    hsv3 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)
    sat3 = hsv3[:, :, 1].mean()
    hue3 = hsv3[:, :, 0].mean()
    blue3 = region3[:, :, 0].mean()
    if sat3 < 76.65:  # Midpoint between com3=56.10 and epic3=97.20
        votes_com3 += 2
    else:
        votes_epic3 += 2
    if hue3 < 96.80:  # Midpoint between com3=77.86 and epic3=115.74
        votes_com3 += 2
    else:
        votes_epic3 += 2
    if blue3 < 97.30:  # Midpoint between com3=80.17 and epic3=114.44
        votes_com3 += 1
    else:
        votes_epic3 += 1
    
    # Region 6: Upper-center (22,9)-(35,16) - 40.0% green diff
    region6 = roi_bgr[9:16, 22:35]
    green6 = region6[:, :, 1].mean()
    hsv6 = cv2.cvtColor(region6, cv2.COLOR_BGR2HSV)
    sat6 = hsv6[:, :, 1].mean()
    if green6 > 121.76:  # Midpoint between com3=146.11 and epic3=97.42
        votes_com3 += 2
    else:
        votes_epic3 += 2
    if sat6 < 77.22:  # Midpoint between com3=63.20 and epic3=91.25
        votes_com3 += 1
    else:
        votes_epic3 += 1
    
    # Region 8: Lower-center (31,26)-(41,40) - 66.3% saturation diff, 52.4% blue diff
    region8 = roi_bgr[26:40, 31:41]
    hsv8 = cv2.cvtColor(region8, cv2.COLOR_BGR2HSV)
    sat8 = hsv8[:, :, 1].mean()
    blue8 = region8[:, :, 0].mean()
    if sat8 < 91.92:  # Midpoint between com3=61.46 and epic3=122.38
        votes_com3 += 3  # Very strong signal
    else:
        votes_epic3 += 3
    if blue8 < 103.18:  # Midpoint between com3=76.13 and epic3=130.23
        votes_com3 += 2
    else:
        votes_epic3 += 2
    
    return 'com3' if votes_com3 > votes_epic3 else 'epic3'

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
    ('com3', 'dirt3'): discriminate_com3_vs_dirt3,
    ('dirt3', 'com3'): discriminate_com3_vs_dirt3,
    ('dirt3', 'epic3'): discriminate_dirt3_vs_epic3,
    ('epic3', 'dirt3'): discriminate_dirt3_vs_epic3,
    ('com3', 'epic3'): discriminate_com3_vs_epic3,
    ('epic3', 'com3'): discriminate_com3_vs_epic3,
    ('leg3', 'com3'): discriminate_leg3_vs_com3,
    ('com3', 'leg3'): discriminate_leg3_vs_com3,
    ('dirt3', 'leg3'): discriminate_dirt3_vs_leg3,
    ('leg3', 'dirt3'): discriminate_dirt3_vs_leg3,
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
