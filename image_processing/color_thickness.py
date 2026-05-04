import cv2
import numpy as np


def extract_wafer_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=100,
        param1=50, param2=30,
        minRadius=100, maxRadius=1000
    )

    mask = np.zeros_like(gray)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(mask, (x,y), r, 255, -1)
        wafer = cv2.bitwise_and(image, image, mask=mask)
        return wafer
    else:
        return image


def remove_background(wafer):
    hsv = cv2.cvtColor(wafer, cv2.COLOR_BGR2HSV)

    # Remove dark + very low saturation (background)
    lower = np.array([0, 40, 40])
    upper = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    clean = cv2.bitwise_and(wafer, wafer, mask=mask)

    return clean


def get_dominant_hsv(image):
    """
    Use median instead of mean (more robust)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1,3)

    # remove dark pixels
    pixels = pixels[pixels[:,2] > 50]

    if len(pixels) == 0:
        return None

    # 🔥 use median instead of mean
    avg = np.median(pixels, axis=0)

    return avg


def map_color_to_thickness(h):
    color_map = [
        ((0, 15), "brown", 700),
        ((15, 30), "yellow/orange", 2000),
        ((30, 60), "yellow-green", 3700),
        ((60, 90), "green", 3500),
        ((90, 120), "blue-green", 5000),
        ((120, 140), "blue", 3100),
        ((140, 170), "violet", 1000),
        ((170, 180), "red-violet", 2500),
    ]

    for (low, high), name, thickness in color_map:
        if low <= h <= high:
            return name, thickness

    return "Unknown", None


def estimate_color_thickness(image):
    wafer = extract_wafer_region(image)
    clean = remove_background(wafer)

    # 🔥 Use center region only (avoid gradient mixing)
    h, w, _ = clean.shape
    center = clean[:, int(w*0.3):int(w*0.7)]

    avg_hsv = get_dominant_hsv(center)

    if avg_hsv is None:
        return None, None, clean

    h_val, s, v = avg_hsv

    color_name, thickness = map_color_to_thickness(h_val)

    return color_name, thickness, clean