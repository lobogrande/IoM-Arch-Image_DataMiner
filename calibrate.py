import pyautogui
import time
from PIL import Image

def calibrate():
    print("--- Game Area Calibration ---")
    print("1. Hover your mouse over the TOP-LEFT corner of the game window.")
    print("   Waiting 5 seconds...")
    time.sleep(5)
    top_left = pyautogui.position()
    print(f"Captured Top-Left: {top_left}")

    print("\n2. Hover your mouse over the BOTTOM-RIGHT corner of the game window.")
    print("   Waiting 5 seconds...")
    time.sleep(5)
    bottom_right = pyautogui.position()
    print(f"Captured Bottom-Right: {bottom_right}")

    # Calculate dimensions
    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y
    
    print("\n--- RESULTS ---")
    print(f"X: {top_left.x}")
    print(f"Y: {top_left.y}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    
    # Take a verification snapshot
    region = (top_left.x, top_left.y, width, height)
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save("calibration_check.png")
    
    print("\nCalibration check saved as 'calibration_check.png'.")
    print("Verify this image shows your game window perfectly!")

if __name__ == "__main__":
    calibrate()