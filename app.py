import streamlit as st
import streamlit.components.v1 as components
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import folium
import requests
from io import BytesIO
from PIL import Image
import os
import json

# Try importing geopy
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Forest Change AI")

def init_ee():
    """Initialize Earth Engine using a token from Streamlit Secrets."""
    try:
        # 1. Try initializing (works if already authenticated locally)
        ee.Initialize()
    except Exception:
        # 2. If that fails (like on Cloud), look for the token in Secrets
        if "earth_engine_token" in st.secrets:
            # Create the hidden folder structure Earth Engine expects
            home_dir = os.path.expanduser("~")
            ee_cred_dir = os.path.join(home_dir, '.config', 'earthengine')
            os.makedirs(ee_cred_dir, exist_ok=True)
            cred_path = os.path.join(ee_cred_dir, 'credentials')
            
            # Write the secret JSON to the credentials file
            with open(cred_path, 'w') as f:
                f.write(st.secrets["earth_engine_token"])
            
            # Try initializing again now that the credentials exist
            try:
                ee.Initialize()
            except Exception as e:
                st.error(f"Authentication failed even with secrets: {e}")
                st.stop()
        else:
            st.error("Earth Engine credentials not found! Please add 'earth_engine_token' to your Streamlit Secrets.")
            st.stop()

# Run the initialization
init_ee()

# --- 2. SIDEBAR ---
st.sidebar.title("🌲 Forest Change Settings")
st.sidebar.subheader("1. Study Area")
lat = st.sidebar.number_input("Latitude", value=22.25, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=85.25, format="%.4f")
radius = st.sidebar.slider("Radius (km)", 1, 50, 10)

st.sidebar.subheader("2. Time Periods")
year_hist = st.sidebar.selectbox("Historical Year", range(2000, 2015), index=5) # 2005
year_recent = st.sidebar.selectbox("Recent Year", range(2015, 2024), index=8)   # 2023

st.sidebar.subheader("3. Processing Settings")
n_clusters = 3 
clean_noise = st.sidebar.checkbox("Apply Noise Filter (Smoother Map)", value=True)
export_scale = st.sidebar.slider("Export Zoom Scope", 1.0, 3.0, 1.2)

custom_title = st.sidebar.text_input("Map Label", placeholder="Auto-detecting...")
run_btn = st.sidebar.button("🚀 Run Unsupervised Analysis")

# --- 3. HELPER FUNCTIONS ---

def get_city_name(lat, lon):
    if not GEOPY_AVAILABLE: return "Unknown Location"
    try:
        geolocator = Nominatim(user_agent="forest_ai_km_final")
        location = geolocator.reverse((lat, lon), language='en', exactly_one=True)
        if location:
            addr = location.raw.get('address', {})
            return addr.get('city') or addr.get('town') or addr.get('village') or "Study Area"
    except:
        return "Unknown Location"
    return "Unknown Location"

def mask_landsat_sr(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(mask)

def get_landsat_collection(year, geometry):
    if year < 2013:
        col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        nir = 'SR_B4'; red = 'SR_B3'
    else:
        col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        nir = 'SR_B5'; red = 'SR_B4'
        
    img = (col.filterBounds(geometry)
           .filterDate(f'{year}-01-01', f'{year}-12-31')
           .filter(ee.Filter.lt('CLOUD_COVER', 15))
           .map(mask_landsat_sr)
           .median()
           .clip(geometry))
    
    return img, nir, red

def add_ee_layer(map_object, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(map_object)
    except:
        pass

def download_url(url):
    response = requests.get(url)
    return BytesIO(response.content)

def add_north_arrow(ax):
    x, y, arrow_length = 0.95, 0.95, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20, xycoords=ax.transAxes)

# --- 4. MAIN LOGIC ---
st.title("🛰️ AI-Enabled Forest Change (Unsupervised)")
st.markdown(f"**Target:** {lat}, {lon} | **Algorithm:** K-Means Clustering")

if 'map_generated' not in st.session_state: st.session_state.map_generated = False
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'auto_label' not in st.session_state: st.session_state.auto_label = ""

if run_btn:
    st.session_state.analysis_done = True
    st.session_state.map_generated = False
    if not custom_title:
        with st.spinner("Detecting location name..."):
            st.session_state.auto_label = get_city_name(lat, lon)
    else:
        st.session_state.auto_label = custom_title

if st.session_state.analysis_done:
    with st.spinner('Running K-Means Clustering...'):
        
        # A. Geometry
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius * 1000).bounds()
        export_aoi = point.buffer(radius * 1000 * export_scale).bounds()

        # B. Get Data
        img_hist, nir_h, red_h = get_landsat_collection(year_hist, aoi)
        img_recent, nir_r, red_r = get_landsat_collection(year_recent, aoi)

        # C. Calculate Indices
        ndvi_hist = img_hist.normalizedDifference([nir_h, red_h]).rename('NDVI_Hist')
        ndvi_recent = img_recent.normalizedDifference([nir_r, red_r]).rename('NDVI_Recent')
        ndvi_diff = ndvi_recent.subtract(ndvi_hist).rename('Change')
        
        # D. Stack for Unsupervised Learning
        stack = ndvi_diff.addBands([ndvi_hist, ndvi_recent])

        # E. K-MEANS CLUSTERING
        training = stack.sample(**{
            'region': aoi, 'scale': 30, 'numPixels': 5000, 'geometries': True
        })
        
        clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
        result_raw = stack.cluster(clusterer)

        # F. NOISE FILTERING
        if clean_noise:
            result = result_raw.focal_mode(1, 'square', 'pixels')
        else:
            result = result_raw

        # G. RELATIVE SORTING LOGIC
        # 1. Calculate Mean Change
        stats_input = ndvi_diff.addBands(result)
        stats = stats_input.reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName='cluster_id'),
            geometry=aoi, scale=100, maxPixels=1e9
        ).getInfo()['groups']

        # 2. Sort clusters (Lowest=Loss, Middle=Stable, Highest=Gain)
        sorted_stats = sorted(stats, key=lambda x: x['mean'])
        
        # 3. Remap
        remap_from = [item['cluster_id'] for item in sorted_stats]
        remap_to = [0, 1, 2] # 0=Loss, 1=Stable, 2=Gain
        
        labels_map = {0: "Degradation/Loss", 1: "Stable", 2: "Regrowth/Gain"}

        # 4. Create Visual Image
        final_vis_image = result.remap(remap_from, remap_to)

        # H. CALCULATE AREAS
        area_stats = ee.Image.pixelArea().addBands(final_vis_image).reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='new_class_id'),
            geometry=aoi, scale=30, maxPixels=1e9
        ).getInfo()['groups']

        final_data = []
        for item in area_stats:
            cid = item['new_class_id']
            sq_km = item['sum'] / 1e6
            final_data.append({"Class": labels_map.get(cid, "Unknown"), "Area (sq km)": sq_km})
        
        df = pd.DataFrame(final_data)

    # --- 5. VISUALIZATION ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Unsupervised Change Map")
        m = folium.Map(location=[lat, lon], zoom_start=11)
        
        ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
        add_ee_layer(m, ndvi_hist, ndvi_vis, f'NDVI {year_hist}')
        add_ee_layer(m, ndvi_recent, ndvi_vis, f'NDVI {year_recent}')
        
        # Change Vis (Red=Loss, White=Stable, Green=Gain)
        change_vis = {'min': 0, 'max': 2, 'palette': ['#ff0000', '#eeeeee', '#00aa00']}
        add_ee_layer(m, final_vis_image, change_vis, 'Forest Change')
        
        folium.LayerControl().add_to(m)
        map_html = m.get_root().render()
        components.html(map_html, height=600)

        # STATIC MAP EXPORT
        st.subheader("🖨️ Export High-Res Map")
        if st.button("Generate Map"):
            st.session_state.map_generated = True
        
        if st.session_state.map_generated:
            with st.spinner("Rendering..."):
                try:
                    vis_params = {'min': 0, 'max': 2, 'palette': ['#ff0000', '#eeeeee', '#00aa00'], 'dimensions': 1000}
                    url = final_vis_image.clip(export_aoi).getThumbURL(vis_params)
                    image_data = download_url(url)
                    pil_img = Image.open(image_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    coords = export_aoi.getInfo()['coordinates'][0]
                    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
                    extent = [min(lons), max(lons), min(lats), max(lats)]
                    
                    ax.imshow(pil_img, extent=extent, aspect='auto')
                    ax.grid(True, color='black', alpha=0.3, linestyle='--')
                    
                    display_name = custom_title if custom_title else st.session_state.auto_label
                    ax.set_title(f"Unsupervised Forest Change ({year_hist}-{year_recent})\n{display_name}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                    
                    add_north_arrow(ax)
                    
                    legend_elements = [
                        Patch(facecolor='#ff0000', label='Degradation/Loss'),
                        Patch(facecolor='#eeeeee', label='Stable'),
                        Patch(facecolor='#00aa00', label='Regrowth/Gain')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right')
                    
                    st.pyplot(fig)
                    
                    img = BytesIO()
                    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
                    st.download_button("Download Image", img, f"KM_Map_{display_name}.png", "image/png")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("📊 Statistics Report")
        
        st.dataframe(df)
        
        if not df.empty:
            fig, ax = plt.subplots()
            colors = {'Degradation/Loss': '#ff0000', 'Stable': '#cccccc', 'Regrowth/Gain': '#00aa00'}
            bar_colors = [colors.get(x, 'blue') for x in df['Class']]
            
            ax.bar(df['Class'], df['Area (sq km)'], color=bar_colors)
            ax.set_ylabel("Area (sq km)")
            st.pyplot(fig)
            
            loss_val = 0
            if 'Degradation/Loss' in df['Class'].values:
                loss_val = df[df['Class'] == 'Degradation/Loss']['Area (sq km)'].values[0]
                
            st.success("✅ Analysis Complete!")
            st.markdown(f"The unsupervised model identified **{loss_val:.2f} sq km** of potential forest loss.")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "stats.csv", "text/csv")