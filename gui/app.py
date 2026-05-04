import streamlit as st
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing.detect_thickness import detect_thickness
from image_processing.color_thickness import estimate_color_thickness

def deal_grove(A, B, t):
    return (-A + np.sqrt(A**2 + 4*B*t)) / 2


st.title("Oxide Thickness Virtual Lab")

# Sidebar
st.sidebar.header("Process Parameters")

oxidation_type = st.sidebar.selectbox("Oxidation Type", ["Dry", "Wet"])
temperature = st.sidebar.number_input("Temperature (°C)", value=1000)
time_hr = st.sidebar.number_input("Oxidation Time (hours)", value=1.0, step=0.1)

if oxidation_type == "Dry":
    A = 100
    B = 11700
else:
    A = 0.05
    B = 0.02

# 🔥 Toggle mode
use_color = st.checkbox("Use Optical (Color-Based) Method")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =============================
    # 🔥 MODE 1: OPTICAL (COLOR)
    # =============================
    if use_color:

        st.header("Color-Based Thickness (Optical)")

        color_name, thickness_color, clean_img = estimate_color_thickness(image)

        st.image(clean_img, caption="Processed Wafer Region")

        if color_name is None:
            st.error("Could not detect color properly")
        else:
            st.write(f"Detected Color: {color_name}")

            if thickness_color:
                st.success(f"Estimated Thickness: {thickness_color} Å")
            else:
                st.warning("Thickness not found for detected color")

    # =============================
    # 🔥 MODE 2: SEM
    # =============================
    else:

        st.header("SEM-Based Thickness")

        result = detect_thickness(image)

        if result is None:
            st.error("❌ Could not detect oxide boundaries. Try another image.")
        else:
            thickness_pixels, top, bottom, nm_per_pixel = result

            image_copy = image.copy()
            cv2.line(image_copy, (0, top), (image.shape[1], top), (0,255,0), 2)
            cv2.line(image_copy, (0, bottom), (image.shape[1], bottom), (0,255,0), 2)

            st.image(image_copy, caption="Detected Interfaces")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Thickness (pixels)", f"{thickness_pixels:.2f}")

            if nm_per_pixel is not None:
                thickness_nm = thickness_pixels * nm_per_pixel

                with col2:
                    st.metric("Thickness (nm)", f"{thickness_nm:.2f}")

                predicted_nm = deal_grove(A, B, time_hr)
                error = abs(thickness_nm - predicted_nm) / predicted_nm * 100

                st.subheader("Model Comparison")

                col3, col4 = st.columns(2)

                with col3:
                    st.metric("Measured (SEM)", f"{thickness_nm:.2f} nm")

                with col4:
                    st.metric("Predicted (Model)", f"{predicted_nm:.2f} nm")

                st.write(f"Error: {error:.2f}%")

            else:
                st.warning("⚠ Scale bar not detected. Thickness only in pixels.")