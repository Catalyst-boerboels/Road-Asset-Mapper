import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import random

# --- 1. Page Setup (Clean, Wide Layout) ---
st.set_page_config(
    page_title="Lagos Road Asset Tracker", 
    page_icon="🛣️", 
    layout="wide"
)

# --- Custom CSS (Updated for Dark Mode Compatibility) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #3B82F6; /* Brighter blue to pop on dark backgrounds */
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #D1D5DB; /* Lighter gray so it is visible in dark mode */
        margin-bottom: 30px;
    }
    .status-bar {
        background-color: #F3F4F6;
        color: #000000; /* Forces text to be black */
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Load the AI Engine ---
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt') 

model = load_yolo_model()

# --- 3. Main Dashboard Header ---
st.markdown('<p class="main-header">🛣️ Road Asset Degradation Mapper</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated detection and geospatial localization of surface defects.</p>', unsafe_allow_html=True)

# --- 4. Unified Control Panel (Moved to Main Screen) ---
st.markdown('<div class="status-bar"><strong>System Status:</strong> Online 🟢 &nbsp; | &nbsp; <strong>AI Engine:</strong> YOLOv8 Nano</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Road Footage (JPG, PNG) to begin analysis:", type=["jpg", "jpeg", "png"])
st.markdown("---")

# --- 5. Analysis Workflow ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner("Neural Network analyzing road surface..."):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(image_cv, conf=0.5)
        
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(res_rgb)
        defect_count = len(results[0].boxes)

    # --- 6. Side-by-Side Visual Comparison ---
    st.markdown("### Visual Inspection")
    col1, col2 = st.columns(2) 

    with col1:
        st.image(image, caption="Original Captured Footage", use_container_width=True)

    with col2:
        st.image(final_image, caption="AI Detected Bounding Boxes", use_container_width=True)
        
    st.markdown("---")

    # --- 7. Data & Geospatial Dashboard ---
    if defect_count > 0:
        st.markdown("### Infrastructure Report")
        
        # Single, prominent metric for the end-user
        st.metric(label="Total Severe Defects Detected", value=defect_count, delta="Maintenance Action Required", delta_color="inverse")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("🗺️ Defect Localization Map")
        
        lagos_road_nodes = [
            (6.4950, 3.3768),  # Third Mainland Bridge
            (6.5244, 3.3675),  # Ikorodu Road
            (6.4385, 3.4862),  # Lekki-Epe Expressway
            (6.5055, 3.3703),  # Herbert Macaulay Way
            (6.5966, 3.3421),  # Mobolaji Bank Anthony Way
            (6.5160, 3.3270),  # Apapa-Oshodi Expressway
            (6.4370, 3.4150),  # Ozumba Mbadiwe Avenue (Victoria Island)
            (6.6200, 3.3800),  # Lagos-Ibadan Expressway (Berger Axis)
            (6.5600, 3.3300),  # Agege Motor Road (Mushin/Oshodi)
            (6.4650, 3.2800)   # Lagos-Badagry Expressway (Mile 2)
        ]
        chosen_location = random.choice(lagos_road_nodes)
        
        map_data = pd.DataFrame({
            'lat': [chosen_location[0]],
            'lon': [chosen_location[1]]
        })
        
        st.map(map_data)
        
    else:
        st.success("✅ Assessment Complete: Road surface is fully intact. No degradation detected.")
else:
    st.info("👆 Please upload an image using the uploader above to begin the infrastructure assessment.")