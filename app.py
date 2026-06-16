import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
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

# --- 3. GPS EXIF EXTRACTION LOGIC ---
def get_exif_data(image):
    """Extracts raw EXIF data from a PIL Image."""
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = ExifTags.GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
    except Exception as e:
        pass # Image might not have EXIF data
    return exif_data

def convert_to_degrees(value):
    """Helper function to convert GPS coordinates to decimal format."""
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(exif_data):
    """Parses latitude and longitude from the EXIF dictionary."""
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get("GPSLatitudeRef")
        gps_longitude = gps_info.get("GPSLongitude")
        gps_longitude_ref = gps_info.get("GPSLongitudeRef")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = convert_to_degrees(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon
            return lat, lon
    return None, None

# --- 4. Main Dashboard Header ---
st.markdown('<p class="main-header">🛣️ Road Asset Degradation Mapper</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated detection and geospatial localization of surface defects.</p>', unsafe_allow_html=True)

# --- 5. Unified Control Panel ---
st.markdown('<div class="status-bar"><strong>System Status:</strong> Online 🟢 &nbsp; | &nbsp; <strong>AI Engine:</strong> YOLOv8 Nano</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Road Footage (JPG, PNG) to begin analysis:", type=["jpg", "jpeg", "png"])
st.markdown("---")

# --- 6. Analysis Workflow ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Extract Live GPS metadata BEFORE converting to cv2 format
    exif_data = get_exif_data(image)
    live_lat, live_lon = get_lat_lon(exif_data)
    
    with st.spinner("Neural Network analyzing road surface..."):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(image_cv, conf=0.5)
        
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(res_rgb)
        defect_count = len(results[0].boxes)

    # --- 7. Side-by-Side Visual Comparison ---
    st.markdown("### Visual Inspection")
    col1, col2 = st.columns(2) 

    with col1:
        st.image(image, caption="Original Captured Footage", use_container_width=True)

    with col2:
        st.image(final_image, caption="AI Detected Bounding Boxes", use_container_width=True)
        
    st.markdown("---")

    # --- 8. Data & Geospatial Dashboard ---
    if defect_count > 0:
        st.markdown("### Infrastructure Report")
        
        # Single, prominent metric for the end-user
        st.metric(label="Total Severe Defects Detected", value=defect_count, delta="Maintenance Action Required", delta_color="inverse")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("🗺️ Defect Localization Map")
        
        # Mapping Logic: Prefer Live GPS, Fallback to Simulated
        if live_lat is not None and live_lon is not None:
            st.success(f"📍 **Live Metadata Detected!** Mapping real coordinates: {live_lat:.5f}, {live_lon:.5f}")
            plot_lat = live_lat
            plot_lon = live_lon
            zoom_level = 15 # Zoom in closer if it's a real live location
        else:
            st.warning("⚠️ No GPS metadata found in image. Using simulated fallback node.")
            lagos_road_nodes = [
                {"name": "Third Mainland Bridge", "lat": 6.4950, "lon": 3.3768},
                {"name": "Ikorodu Road", "lat": 6.5244, "lon": 3.3675},
                {"name": "Lekki-Epe Expressway", "lat": 6.4385, "lon": 3.4862},
                {"name": "Herbert Macaulay Way", "lat": 6.5055, "lon": 3.3703},
                {"name": "Mobolaji Bank Anthony Way", "lat": 6.5966, "lon": 3.3421},
                {"name": "Apapa-Oshodi Expressway", "lat": 6.5160, "lon": 3.3270},
                {"name": "Ozumba Mbadiwe Avenue (Victoria Island)", "lat": 6.4370, "lon": 3.4150},
                {"name": "Lagos-Ibadan Expressway (Berger Axis)", "lat": 6.6200, "lon": 3.3800},
                {"name": "Agege Motor Road (Mushin/Oshodi)", "lat": 6.5600, "lon": 3.3300},
                {"name": "Lagos-Badagry Expressway (Mile 2)", "lat": 6.4650, "lon": 3.2800}
            ]
            chosen_location = random.choice(lagos_road_nodes)
            st.write(f"**Simulated Location:** {chosen_location['name']}")
            plot_lat = chosen_location['lat']
            plot_lon = chosen_location['lon']
            zoom_level = 11 # Zoom out for a broader city view
        
        map_data = pd.DataFrame({
            'lat': [plot_lat],
            'lon': [plot_lon]
        })
        
        st.map(map_data, zoom=zoom_level)
        
    else:
        st.success("✅ Assessment Complete: Road surface is fully intact. No degradation detected.")
else:
    st.info("👆 Please upload an image using the uploader above to begin the infrastructure assessment.")
