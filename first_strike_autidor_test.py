def get_grid_dna_v16_1(img_gray):
    """
    Samples the BOTTOM-RIGHT of each slot. 
    The Master Noise Profile proves this is the only zone safe from damage text.
    """
    dna = ""
    for slot in range(24):
        row, col = divmod(slot, 6)
        # Shift sample to BOTTOM-RIGHT of the 48x48 block (+15, +15 from center)
        cx = int(SLOT1_CENTER[0] + (col * STEP_X)) + 15
        cy = int(SLOT1_CENTER[1] + (row * STEP_Y)) + 15
        
        # Guard against edge-of-screen coordinate overflow
        if cy >= img_gray.shape[0] or cx >= img_gray.shape[1]:
            dna += "0"
            continue
            
        roi = img_gray[cy-4:cy+4, cx-4:cx+4]
        # Using a slightly higher mean threshold to distinguish ground texture from ore shadow
        dna += "1" if np.mean(roi) > 58 else "0"
    return dna