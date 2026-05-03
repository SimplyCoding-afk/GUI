import numpy as np

def deal_grove(A, B, t):
    return (-A + np.sqrt(A**2 + 4*B*t)) / 2

import streamlit as st
import numpy as np
import cv2
import sys
import os

# allow import from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing.detect_thickness import detect_thickness
from image_processing.calibration import load_calibration

st.title("Oxide Thickness Virtual Lab")

st.sidebar.header("Process Parameters")

oxidation_type = st.sidebar.selectbox("Oxidation Type", ["Dry", "Wet"])
temperature = st.sidebar.number_input("Temperature (°C)", value=1000)
time_hr = st.sidebar.number_input("Oxidation Time (hours)", value=1.0, step=0.1)

# ---- NOW define constants ----
if oxidation_type == "Dry":
    A = 100
    B = 11700
else:
    A = 50
    B = 40000

uploaded_file = st.file_uploader("Upload SEM Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show original image
    st.image(image, caption="Uploaded Image", width=600)

    # -------- DETECTION --------
    thickness_pixels, top, bottom, nm_per_pixel = detect_thickness(image)

    # Draw detected boundaries
    image_copy = image.copy()
    cv2.line(image_copy, (0, top), (image.shape[1], top), (0,255,0), 2)
    cv2.line(image_copy, (0, bottom), (image.shape[1], bottom), (0,255,0), 2)

    cv2.putText(image_copy, "Top", (10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(image_copy, "Bottom", (10, bottom+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    st.image(image_copy, caption="Detected Boundaries")

    # -------- DISPLAY RESULTS --------
    st.success(f"Thickness: {thickness_pixels:.2f} pixels")

    if nm_per_pixel is not None:
        thickness_nm = thickness_pixels * nm_per_pixel
        st.success(f"Thickness: {thickness_nm:.2f} nm")

        # -------- Deal-Grove Prediction --------
        predicted_nm = deal_grove(A, B, time_hr)

        error = abs(thickness_nm - predicted_nm) / thickness_nm * 100

        st.subheader("Model Comparison")
        st.write(f"Predicted Thickness: {predicted_nm:.2f} nm")
        st.write(f"Error: {error:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Measured (SEM)", f"{thickness_nm:.2f} nm")

        with col2:
            st.metric("Predicted (Model)", f"{predicted_nm:.2f} nm")

