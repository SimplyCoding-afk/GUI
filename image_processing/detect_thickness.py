import cv2
import numpy as np

from image_processing.calibration import (
    detect_scale_bar,
    load_calibration
)

def detect_thickness(image):

    original_image = image.copy()

    # ---------------- CALIBRATION ----------------
    nm_per_pixel = None

    has_scale = detect_scale_bar(original_image)

    # ⚠️ Disable manual input for Streamlit
    if has_scale:
        nm_per_pixel = load_calibration()
    else:
        nm_per_pixel = load_calibration()

    # ---------------- PREPROCESS ----------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    h, w = gray.shape

    # ---------------- REMOVE METADATA ----------------
    bottom_region = gray[int(h*0.85):h, :]
    if np.mean(bottom_region) < 40:
        gray = gray[:int(h*0.85), :]

    h, w = gray.shape

    # ---------------- MULTI-COLUMN DETECTION ----------------
    num_samples = 120

    columns = np.linspace(int(w*0.1), int(w*0.9), num_samples).astype(int)

    top_edges = []
    bottom_edges = []

    for col in columns:

        profile = gray[:, col]
        gradient = np.gradient(profile)

        top = np.argmax(gradient)

        # 🔒 SAFETY CHECKS (fix your crash)
        if top + 20 >= len(gradient):
            continue

        search = gradient[top+20:]

        if len(search) == 0:
            continue

        bottom = top + 20 + np.argmin(search)

        top_edges.append(top)
        bottom_edges.append(bottom)

    # 🔒 FINAL SAFETY
    if len(top_edges) == 0 or len(bottom_edges) == 0:
        return 0, 0, 0, None

    top_edges = np.array(top_edges)
    bottom_edges = np.array(bottom_edges)

    thickness_pixels = np.mean(bottom_edges - top_edges)

    avg_top = int(np.mean(top_edges))
    avg_bottom = int(np.mean(bottom_edges))

    return thickness_pixels, avg_top, avg_bottom, nm_per_pixel