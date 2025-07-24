# 🌱 Agri-Smart: Agricultural Advisory System

Agri-Smart is an intelligent agricultural advisory platform designed to assist farmers and agronomists in making data-driven decisions. It leverages machine learning to deliver personalized crop recommendations, detect crop diseases from images, and provide rich data visualizations for agricultural insights.

---

## 📋 Table of Contents

* [Features](#features)
* [Demo](#demo)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)
* [Dataset](#dataset)
* [Contributing](#contributing)

---

## 🚀 Features

* **Crop Recommendation**: Suggests suitable crops based on soil nutrients (N, P, K), climate conditions (temperature, humidity, rainfall), and advanced parameters (salinity, water requirement, disease resistance).
* **Disease Identification**: Detects plant diseases from uploaded crop images using a KNN-based image classification model with treatment/prevention advice.
* **Data Insights**: Visualize crop distributions, feature importance, correlations, and model performance via interactive plots and ROC curves.
* **Rabi & Kharif Crop Advisory**: Seasonal recommendations and best practices for Indian cropping patterns.
* **Modern UI**: Clean, responsive interface with custom green-themed aesthetics and smooth navigation.

---

## 📺 Demo

To run the app locally:

```bash
streamlit run app.py --server.port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Subhapreet21/Agri-Smart.git
cd AGRI-SMART
```

### 2. Install dependencies

> Recommended: Python 3.11+

```bash
pip install -r requirements.txt
```

Alternatively, install via:

```bash
pip install -r pyproject.toml
```

---

## 📌 Usage

### 🌾 Crop Recommendation

* Navigate to the **Crop Recommendation** tab.
* Fill in the required parameters:

  * N, P, K
  * Temperature, Humidity, pH, Rainfall
* Expand **Advanced Parameters** for:

  * Salinity
  * Water Requirement
  * Disease Resistance Score
* Click **Get Recommendation** to see the top 3 crop suggestions with confidence scores.

### 🌿 Disease Detection

* Go to the **Disease Detection** tab.
* Upload a crop image (`.jpg`, `.jpeg`, or `.png`).
* Click **Analyze Image** to detect the disease and receive recommendations.

### 📊 Data Insights

* Navigate to **Data Insights** for:

  * Crop distribution
  * Parameter-based analysis
  * Feature importance
  * Correlation heatmaps
  * ROC curve visualizations

### 🗓 Seasonal Crop Advice

* Visit **Rabi Crops** or **Kharif Crops** for seasonal tips and best agricultural practices.

---

## 📂 Project Structure

```
AGRI-SMART/
├── app.py                       # Main Streamlit app
├── crop_recommendation.py      # ML logic for crop suggestion
├── disease_identification.py   # KNN model for disease detection
├── data_visualization.py       # Chart and plot generation
├── utils.py                    # Data loading/preprocessing helpers
├── models/                     # Trained model files (.pkl)
├── assets/                     # Custom CSS and images
├── crop_dataset.csv            # Main dataset
├── train_disease_knn_model.py  # Script to train/save disease model
├── pyproject.toml              # Python dependencies
└── .streamlit/                 # Streamlit configuration
```

---

## 📦 Dependencies

Install all packages using:

```bash
pip install -r requirements.txt
```

### Python Packages

* `streamlit`
* `streamlit-option-menu`
* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `plotly`
* `Pillow`
* `joblib`
* `pickle-mixin`

---

## 📊 Dataset

* **crop\_dataset.csv** — Contains features like:

  * Macronutrients: N, P, K
  * Environmental: Temperature, Humidity, pH, Rainfall
  * Advanced: Salinity, Water Requirement, Disease Resistance
  * Label: Crop Name and Common Diseases

This dataset powers both the recommendation engine and the visualization dashboards.

---

## 🤝 Contributing

We welcome contributions! Feel free to:

* Open an issue for bugs or suggestions
* Submit a pull request for new features or improvements
* Fork the repo and build your own variant

---
