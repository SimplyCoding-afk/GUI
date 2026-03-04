import cv2
import numpy as np
import json
import os

CALIBRATION_FILE = "calibration.json"


# ---------------------------------------------------
# Detect if scale bar likely exists
# ---------------------------------------------------
def detect_scale_bar(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bottom = gray[int(h*0.80):h, :]

    edges = cv2.Canny(bottom, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=10
    )

    return lines is not None


# ---------------------------------------------------
# Manual scale bar calibration
# ---------------------------------------------------
def calculate_nm_per_pixel(image):

    print("\nScale bar detected.")
    scale_nm = float(input("Enter scale bar value (nm): "))

    h, w = image.shape[:2]

    crop_start = int(h*0.80)
    cropped = image[crop_start:h, :]

    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y+crop_start))
            print("Point:", points[-1])

    cv2.namedWindow("Select Scale Bar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Scale Bar", click)

    print("Click LEFT and RIGHT ends of the scale bar")

    while True:
        cv2.imshow("Select Scale Bar", cropped)

        if len(points) == 2:
            break

        cv2.waitKey(1)

    cv2.destroyAllWindows()

    p1, p2 = points

    pixel_distance = np.sqrt(
        (p1[0]-p2[0])**2 +
        (p1[1]-p2[1])**2
    )

    nm_per_pixel = scale_nm / pixel_distance

    print("Scale bar pixels:", pixel_distance)
    print("nm per pixel:", nm_per_pixel)

    save_calibration(nm_per_pixel, scale_nm, pixel_distance)

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