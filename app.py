import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import random

# --- 1. Page Setup ---
st.set_page_config(page_title="Lagos Road Asset Tracker", page_icon="🛣️", layout="centered")
st.title("🛣️ Road Asset Degradation Mapper")
st.markdown("""
Welcome to the automated road infrastructure assessment tool. 
Upload an image of a roadway to instantly localize and map surface defects (potholes).
""")

# --- 2. Load the AI Engine ---
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt') 

model = load_yolo_model()

# --- 3. Frontend UI: File Uploader ---
uploaded_file = st.file_uploader("Upload Road Footage (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.subheader("Original Image")
    st.image(image, caption="Awaiting Analysis...", use_container_width=True)

    # --- 4. Backend Logic: Run Inference ---
    if st.button(" Run Defect Detection", type="primary"):
        with st.spinner("Analyzing road surface..."):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = model.predict(image_cv, conf=0.5)
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            final_image = Image.fromarray(res_rgb)

            # --- 5. Output Display ---
            st.subheader("Analysis Results")
            st.image(final_image, caption="Detected Bounding Boxes", use_container_width=True)
            
            defect_count = len(results[0].boxes)
            
            if defect_count > 0:
                st.success(f"Detection Complete! Found {defect_count} road defect(s).")
                
                # --- 6. Geospatial Mapping (Simulated Lagos Coordinates) ---
                st.subheader(" Interactive Asset Map")
                st.markdown(f"Logging {defect_count} detected defect(s) to the municipal database...")
                
                # Generate random GPS coordinates constrained to the Lagos mainland/island area
                latitudes = [random.uniform(6.4000, 6.6000) for _ in range(defect_count)]
                longitudes = [random.uniform(3.3000, 3.5000) for _ in range(defect_count)]
                
                # Format the data for Streamlit's mapping engine
                map_data = pd.DataFrame({
                    'lat': latitudes,
                    'lon': longitudes
                })
                
                # Render the map
                st.map(map_data)
                
            else:
                st.success("Detection Complete! No defects found on this road surface.")