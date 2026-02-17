
# 🚓 PatrolIQ – Smart Safety Analytics Platform

## Project Overview
PatrolIQ is an urban safety intelligence platform that uses **unsupervised machine learning** to analyze crime patterns and optimize police patrol allocation.  
The project is built using the **Chicago Crime Dataset** and delivers actionable insights through clustering, dimensionality reduction, MLflow tracking, and a Streamlit web app.

---

## 🎯 Problem Statement
Law enforcement agencies face challenges in:
- Identifying crime hotspots
- Understanding temporal crime patterns
- Allocating patrol resources efficiently

PatrolIQ solves this by transforming large-scale crime data into **visual, data-driven insights**.

---

## 📊 Dataset
- **Source:** Chicago Data Portal – Crimes 2001 to Present
- **Records Used:** 500,000 (sampled)
- **Features:** 22+ crime, temporal, and geographic variables

---

## 🛠️ Project Architecture

```
Raw Crime Data
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Clustering (KMeans, DBSCAN, Hierarchical)
      ↓
Dimensionality Reduction (PCA, t-SNE)
      ↓
MLflow Experiment Tracking
      ↓
Streamlit Application
```

---

## 📁 Project Structure

```
PatrolIQ/
├── PatrolIQ_Data_Cleaning_Preprocessing.ipynb
├── PatrolIQ_Clustering.ipynb
├── PatrolIQ_PCA_TSNE.ipynb
├── PatrolIQ_MLflow.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔍 Key Modules

### 1. Data Cleaning & Preprocessing
- Missing value handling
- Duplicate removal
- Datetime parsing
- Feature engineering (Hour, Day, Month, Weekend)
- Scaling & Encoding

### 2. Clustering Analysis
- **KMeans:** Crime hotspot identification
- **DBSCAN:** Density-based clustering with noise detection
- **Hierarchical:** Zone hierarchy analysis
- **Evaluation:** Silhouette Score

### 3. Dimensionality Reduction
- **PCA:** Feature compression with variance preservation
- **t-SNE:** 2D visualization of complex crime patterns

### 4. MLflow Integration
- Experiment tracking
- Parameter logging
- Model versioning

### 5. Streamlit Application
- Interactive crime hotspot map
- Real-time clustering visualization
- Cloud-deployable interface

---

## 📈 Business Impact
- Optimized patrol deployment
- Identification of high-risk zones
- Improved response planning
- Evidence-based decision making

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

---

## ✅ Evaluation Checklist Coverage
- ✔ Data Cleaning & Preprocessing
- ✔ 3 Unsupervised Clustering Algorithms
- ✔ PCA & t-SNE
- ✔ MLflow Experiment Tracking
- ✔ Streamlit Cloud Deployment Ready

---

## 👤 Author
**PatrolIQ – Capstone Project**  
Domain: Public Safety & Urban Analytics


https://github.com/padhma4782/patroliq
https://patroliq-c5najmrf7khbssbqx2qfwt.streamlit.app/
