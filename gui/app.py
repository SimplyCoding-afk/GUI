import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing.detect_thickness import detect_thickness
from image_processing.color_thickness import estimate_color_thickness

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Oxide Thickness Virtual Lab",
    page_icon="🧪",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0B1020;
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DEAL GROVE MODEL
# ---------------------------------------------------
def deal_grove(A, B, t):
    return (-A + np.sqrt(A**2 + 4*B*t)) / 2


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧪 Oxide Thickness Virtual Laboratory")

st.markdown("""
### Interactive Experiment

This virtual laboratory performs:

- 🔬 SEM-based oxide thickness detection
- 🌈 Optical color-based oxide estimation
- 📈 Deal–Grove theoretical comparison
""")

# ---------------------------------------------------
# THEORY SECTION
# ---------------------------------------------------
with st.expander("📘 Theory : Deal–Grove Oxidation Model"):

    st.write("""
The Deal–Grove model describes thermal oxidation of silicon.

### Fundamental Equation

x² + Ax = Bt

Where:

- x = oxide thickness
- A = linear rate constant
- B = parabolic rate constant
- t = oxidation time

Initially oxidation is reaction-controlled (linear),
then becomes diffusion-controlled (parabolic).
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙ Experiment Setup")

analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["SEM Analysis", "Optical Analysis"]
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=1000
)

time_hr = st.sidebar.number_input(
    "Oxidation Time (hours)",
    value=1.0,
    step=0.1
)

# ---------------------------------------------------
# DEAL GROVE PARAMETERS
# ---------------------------------------------------
st.sidebar.subheader("📈 Deal–Grove Parameters")

preset = st.sidebar.selectbox(
    "Oxidation Process Preset",
    [
        "Dry Oxidation (1000°C)",
        "Wet Oxidation (1100°C)",
        "Custom"
    ]
)

# ---------------------------------------------------
# PRESET VALUES
# ---------------------------------------------------
if preset == "Dry Oxidation (1000°C)":

    default_A = 100.0
    default_B = 11700.0

elif preset == "Wet Oxidation (1100°C)":

    default_A = 50.0
    default_B = 80000.0

else:

    default_A = 100.0
    default_B = 11700.0

# ---------------------------------------------------
# USER INPUT PARAMETERS
# ---------------------------------------------------
A = st.sidebar.number_input(
    "Linear Constant A (nm)",
    value=default_A,
    step=10.0
)

B = st.sidebar.number_input(
    "Parabolic Constant B (nm²/hr)",
    value=default_B,
    step=1000.0
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload SEM / Optical Wafer Image",
    type=["png", "jpg", "jpeg"]
)

# ===================================================
# MAIN APPLICATION
# ===================================================
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # ------------------------------------------------
    # TABS
    # ------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🧪 Experiment",
        "🔬 Analysis",
        "📈 Results"
    ])

    # =================================================
    # TAB 1 : EXPERIMENT
    # =================================================
    with tab1:

        st.subheader("Uploaded Sample")

        st.image(
            image,
            use_container_width=True
        )

        st.info("""
### Experiment Workflow

1️⃣ Upload wafer / SEM image  
2️⃣ Preprocess image  
3️⃣ Detect interfaces or optical color  
4️⃣ Calculate oxide thickness  
5️⃣ Compare with Deal–Grove model
""")

    # =================================================
    # TAB 2 : ANALYSIS
    # =================================================
    with tab2:

        # =============================================
        # SEM ANALYSIS
        # =============================================
        if analysis_mode == "SEM Analysis":

            st.header("🔬 SEM Thickness Detection")

            with st.spinner("Processing SEM image..."):

                result = detect_thickness(image)

            if result is None:

                st.error("❌ Could not detect oxide interfaces.")

            else:

                thickness_pixels, top, bottom, nm_per_pixel = result

                # ----------------------------------------
                # Draw boundaries
                # ----------------------------------------
                image_copy = image.copy()

                cv2.line(
                    image_copy,
                    (0, top),
                    (image.shape[1], top),
                    (0,255,0),
                    2
                )

                cv2.line(
                    image_copy,
                    (0, bottom),
                    (image.shape[1], bottom),
                    (0,255,0),
                    2
                )

                st.image(
                    image_copy,
                    caption="Detected Oxide Interfaces",
                    use_container_width=True
                )

                # ----------------------------------------
                # Metrics
                # ----------------------------------------
                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Thickness (pixels)",
                        f"{thickness_pixels:.2f}"
                    )

                if nm_per_pixel is not None:

                    thickness_nm = thickness_pixels * nm_per_pixel

                    with col2:

                        st.metric(
                            "Thickness (nm)",
                            f"{thickness_nm:.2f}"
                        )

                    st.success("SEM analysis completed successfully.")

                else:

                    thickness_nm = None

                    st.warning(
                        "⚠ Calibration unavailable. Thickness shown only in pixels."
                    )

        # =============================================
        # OPTICAL ANALYSIS
        # =============================================
        else:

            st.header("🌈 Optical Color-Based Thickness")

            with st.spinner("Analyzing wafer color..."):

                color_name, thickness_color, clean_img = estimate_color_thickness(image)

            st.image(
                clean_img,
                caption="Processed Wafer Region",
                use_container_width=True
            )

            if color_name is None:

                st.error("❌ Could not detect wafer color.")

            else:

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Detected Color",
                        color_name
                    )

                with col2:

                    if thickness_color:

                        st.metric(
                            "Estimated Thickness",
                            f"{thickness_color} Å"
                        )

                    else:

                        st.metric(
                            "Estimated Thickness",
                            "Unknown"
                        )

                st.success("Optical analysis completed successfully.")

    # =================================================
    # TAB 3 : RESULTS
    # =================================================
    with tab3:

        st.header("📈 Deal–Grove Model Comparison")

        predicted_nm = deal_grove(A, B, time_hr)

        # ------------------------------------------------
        # Generate Model Curve
        # ------------------------------------------------
        t = np.linspace(0, 10, 300)

        x_curve = deal_grove(A, B, t)

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(
            t,
            x_curve,
            linewidth=2,
            label="Deal–Grove Model"
        )

        # =============================================
        # SEM COMPARISON
        # =============================================
        if analysis_mode == "SEM Analysis":

            if 'thickness_nm' in locals() and thickness_nm is not None:

                ax.scatter(
                    time_hr,
                    thickness_nm,
                    s=120,
                    label="SEM Measurement"
                )

                error = abs(
                    thickness_nm - predicted_nm
                ) / predicted_nm * 100

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Measured SEM Thickness",
                        f"{thickness_nm:.2f} nm"
                    )

                with col2:

                    st.metric(
                        "Model Prediction",
                        f"{predicted_nm:.2f} nm"
                    )

                st.metric(
                    "SEM vs Model Error",
                    f"{error:.2f}%"
                )

        # =============================================
        # OPTICAL COMPARISON
        # =============================================
        else:

            if 'thickness_color' in locals() and thickness_color is not None:

                # Å → nm
                thickness_color_nm = thickness_color / 10

                ax.scatter(
                    time_hr,
                    thickness_color_nm,
                    s=120,
                    label="Optical Measurement"
                )

                error_optical = abs(
                    thickness_color_nm - predicted_nm
                ) / predicted_nm * 100

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Optical Thickness",
                        f"{thickness_color_nm:.2f} nm"
                    )

                with col2:

                    st.metric(
                        "Model Prediction",
                        f"{predicted_nm:.2f} nm"
                    )

                st.metric(
                    "Optical vs Model Error",
                    f"{error_optical:.2f}%"
                )

        # ------------------------------------------------
        # Model prediction point
        # ------------------------------------------------
        ax.scatter(
            time_hr,
            predicted_nm,
            s=120,
            label="Model Prediction"
        )

        # ------------------------------------------------
        # Labels
        # ------------------------------------------------
        ax.set_xlabel("Oxidation Time (hours)")
        ax.set_ylabel("Oxide Thickness (nm)")
        ax.set_title("Oxide Growth using Deal–Grove Model")

        ax.grid(True)

        ax.legend()

        st.pyplot(fig)

        # ------------------------------------------------
        # INTERPRETATION
        # ------------------------------------------------
        st.info("""
### Interpretation

- Thin oxide follows linear growth
- Thick oxide follows parabolic growth
- SEM and optical measurements are compared with Deal–Grove predictions
- Optical estimation uses thin-film interference colors
""")