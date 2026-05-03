import cv2
import numpy as np
import json
import os

CALIBRATION_FILE = "calibration.json"


# ---------------------------------------------------
# Detect if scale bar likely exists
# ---------------------------------------------------
def detect_scale_bar(image):
    return True   # assume scale bar always exists (for now)


# ---------------------------------------------------
# Manual scale bar calibration
# ---------------------------------------------------
def calculate_nm_per_pixel(image):

    # Your SEM image scale bar = 100 nm
    scale_nm = 100  

    h, w = image.shape[:2]

    # Approximate scale bar width (you can tune this later)
    scale_bar_pixels = int(w * 0.18)

    nm_per_pixel = scale_nm / scale_bar_pixels

    return nm_per_pixel


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
# Load previous calibration
# ---------------------------------------------------
def load_calibration():

    if not os.path.exists(CALIBRATION_FILE):
        return None

    with open(CALIBRATION_FILE, "r") as f:
        data = json.load(f)

    return data["nm_per_pixel"]