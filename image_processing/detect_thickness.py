import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration import (
    detect_scale_bar,
    calculate_nm_per_pixel,
    load_calibration
)

# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------

image_path = "image_processing/test_images/sample4.png"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

original_image = image.copy()

# ---------------------------------------------------
# CALIBRATION LOGIC
# ---------------------------------------------------

nm_per_pixel = None

has_scale = detect_scale_bar(original_image)

if has_scale:

    nm_per_pixel = calculate_nm_per_pixel(original_image)

else:

    print("\nNo scale bar detected.")

    prev = load_calibration()

    if prev is not None:

        choice = input("Use previous calibration? (y/n): ")

        if choice.lower() == "y":
            nm_per_pixel = prev
            print("Using previous calibration:", nm_per_pixel)

# ---------------------------------------------------
# PREPARE IMAGE FOR ANALYSIS
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian smoothing
gray = cv2.GaussianBlur(gray, (5,5), 0)

h, w = gray.shape

# ---------------------------------------------------
# Automatic metadata detection
# ---------------------------------------------------

bottom = gray[int(h*0.85):h, :]
mean_intensity = np.mean(bottom)

if mean_intensity < 40:
    print("Metadata detected → cropping bottom region")
    gray = gray[:int(h*0.85), :]
else:
    print("No metadata detected")

h, w = gray.shape

# ---------------------------------------------------
# MULTI COLUMN THICKNESS DETECTION
# ---------------------------------------------------

num_samples = 120

columns = np.linspace(
    int(w*0.1),
    int(w*0.9),
    num_samples
).astype(int)

top_edges = []
bottom_edges = []

for col in columns:

    profile = gray[:, col]

    gradient = np.gradient(profile)

    top = np.argmax(gradient)

    search = gradient[top+20:]
    bottom = top + 20 + np.argmin(search)

    top_edges.append(top)
    bottom_edges.append(bottom)

top_edges = np.array(top_edges)
bottom_edges = np.array(bottom_edges)

thickness_pixels = bottom_edges - top_edges

# ---------------------------------------------------
# STATISTICS
# ---------------------------------------------------

mean_px = np.mean(thickness_pixels)
std_px = np.std(thickness_pixels)

print("\n---- Thickness Statistics ----")
print("Mean thickness (pixels):", round(mean_px,2))
print("Std deviation:", round(std_px,2))
print("Min:", np.min(thickness_pixels))
print("Max:", np.max(thickness_pixels))

# ---------------------------------------------------
# CONVERT TO NM
# ---------------------------------------------------

if nm_per_pixel is not None:

    thickness_nm = mean_px * nm_per_pixel

    print("\n---- Final Result ----")
    print("Thickness:", round(thickness_nm,2), "nm")

else:

    print("\nNo calibration available.")
    print("Thickness:", round(mean_px,2), "pixels")

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------

image_display = original_image.copy()

avg_top = int(np.mean(top_edges))
avg_bottom = int(np.mean(bottom_edges))

cv2.line(image_display, (0,avg_top), (w,avg_top), (0,255,0), 2)
cv2.line(image_display, (0,avg_bottom), (w,avg_bottom), (0,255,0), 2)

plt.figure(figsize=(8,4))
plt.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
plt.title("Detected Oxide Thickness")
plt.axis("off")
plt.show()