import cv2
import numpy as np
import json
import os

CALIBRATION_FILE = "calibration.json"


# ---------------------------------------------------
# Detect if scale bar likely exists (simple check)
# ---------------------------------------------------
def detect_scale_bar(image):
    h, w = image.shape[:2]

    # Assume bottom region contains scale bar
    bottom = image[int(h*0.85):h, :]

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

    # detect bright horizontal region
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    white_ratio = np.sum(thresh == 255) / thresh.size

    return white_ratio > 0.01   # simple heuristic


# ---------------------------------------------------
# Estimate nm/pixel (fallback)
# ---------------------------------------------------
def calculate_nm_per_pixel(image):

    scale_nm = 100  # assumed scale bar value

    h, w = image.shape[:2]

    # assume scale bar ~18% of width (approximation)
    scale_pixels = int(w * 0.18)

    nm_per_pixel = scale_nm / scale_pixels

    return nm_per_pixel, scale_nm, scale_pixels


# ---------------------------------------------------
# Save calibration
# ---------------------------------------------------
def save_calibration(nm_per_pixel, scale_nm, scale_pixels):

    data = {
        "nm_per_pixel": nm_per_pixel,
        "scale_bar_nm": scale_nm,
        "scale_bar_pixels": scale_pixels
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------
# Load or compute calibration
# ---------------------------------------------------
def load_calibration(image=None):

    # Try loading existing calibration
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        return data["nm_per_pixel"]

    # If not available → compute new
    if image is not None:
        nm_per_pixel, scale_nm, scale_pixels = calculate_nm_per_pixel(image)

        save_calibration(nm_per_pixel, scale_nm, scale_pixels)

        return nm_per_pixel

    return None