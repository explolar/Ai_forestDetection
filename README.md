# 🌲 AI-Enabled Forest Change Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aiforestdetection-xwmbbzvquwbkwetrjgf2e7.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **"I spent my weekend teaching AI to find trees so I didn't have to."**

An automated geospatial web application that detects forest cover change (deforestation and regrowth) using **Unsupervised Machine Learning** and **Google Earth Engine**.



## 🚀 Live Demo
👉 **[Click here to try the App](https://aiforestdetection-xwmbbzvquwbkwetrjgf2e7.streamlit.app/)**

---

## 🧐 About The Project

Traditional forest monitoring requires manual digitization or supervised training data (drawing thousands of polygons). This project automates the process using **Unsupervised Learning (K-Means Clustering)**.

The app ingests multi-temporal Landsat satellite imagery, calculates spectral indices, and automatically groups pixels into three categories: **Loss**, **Stable**, and **Gain**.

### Key Features
* **🌍 Global Scale:** Analyze any location on Earth using the Google Earth Engine data catalog.
* **🤖 Unsupervised AI:** Uses K-Means Clustering—no manual training data required.
* **☁️ Auto-Cloud Masking:** Filters out clouds and shadows from Landsat 5, 7, 8, and 9 imagery.
* **📊 Dynamic Thresholding:** Implements relative sorting logic to distinguish "Loss" from "Stable" even in subtle change scenarios.
* **📍 Smart Export:** Generates publication-ready static maps with grids, scale, and north arrows using Matplotlib.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Geospatial Engine:** [Google Earth Engine (GEE)](https://earthengine.google.com/)
* **Data Processing:** Python, Pandas, NumPy
* **Visualization:** Folium (Interactive), Matplotlib (Static)
* **Geocoding:** Geopy

---

## ⚙️ How It Works

1.  **Data Ingestion:** User selects a location and two time periods (Historical vs. Recent). The app fetches Landsat Surface Reflectance data.
2.  **Preprocessing:** Clouds and shadows are masked using the QA_PIXEL band.
3.  **Feature Engineering:**
    * Calculates **NDVI** (Normalized Difference Vegetation Index) for both periods.
    * Computes the difference: $\Delta NDVI = NDVI_{recent} - NDVI_{historical}$
4.  **Clustering:** A K-Means algorithm (k=3) clusters the pixels based on spectral change.
5.  **Labeling:** Clusters are sorted by mean change value:
    * Lowest Mean → **Degradation/Loss** (Red)
    * Middle Mean → **Stable** (White)
    * Highest Mean → **Regrowth/Gain** (Green)

---

## 💻 Local Installation

To run this app locally on your machine:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Ai_Forest_Detection.git](https://github.com/YOUR_USERNAME/Ai_Forest_Detection.git)
    cd Ai_Forest_Detection
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Authenticate Earth Engine**
    Make sure you have a Google Earth Engine account.
    ```bash
    earthengine authenticate
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🔐 Deployment (Streamlit Cloud)

To deploy this on Streamlit Cloud, you must provide your Earth Engine credentials via **Streamlit Secrets**.

1.  Get your refresh token from your local machine:
    * Windows: `C:\Users\YOUR_USER\.config\earthengine\credentials`
    * Mac/Linux: `~/.config/earthengine/credentials`
2.  In Streamlit Cloud settings, add the following secret:

```toml
[earth_engine]
token = '{"refresh_token": "YOUR_TOKEN_HERE", ...}' 
# Copy the full JSON content from your credentials file
