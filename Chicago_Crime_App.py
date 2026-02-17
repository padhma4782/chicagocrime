# Chicago Crime Analytics - Production Cloud Safe Version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(page_title="Chicago Crime Analytics", layout="wide")
st.title("🚓 Chicago Crime Analytics Dashboard")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PatrolIQ_cleaned.csv")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    return df

df = load_data()

# Chicago geographic bounds
lat_min, lat_max = 41.6, 42.1
lon_min, lon_max = -88.0, -87.5

df = df[
    df['Latitude'].between(lat_min, lat_max) &
    df['Longitude'].between(lon_min, lon_max)
].copy()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Crime Zones (K-Means)",
        "Geographic Crime Heatmap",
        "Temporal Pattern Analysis",
        "Dimensionality Reduction (PCA & t-SNE)"
    ]
)

# ==================================================
# 1️⃣ Crime Zones (K-Means)
# ==================================================
if page == "Crime Zones (K-Means)":

    st.subheader("Crime Zones (K-Means Clustering)")

    coords = df[['Latitude', 'Longitude']].dropna()

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    k = st.slider("Number of clusters", 4, 12, 8)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_scaled)

    df_k = coords.copy()
    df_k["zone"] = labels

    fig = px.scatter_mapbox(
        df_k,
        lat="Latitude",
        lon="Longitude",
        color="zone",
        zoom=10,
        height=600,
        mapbox_style="carto-positron"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 2️⃣ Geographic Crime Heatmap (DBSCAN)
# ==================================================
elif page == "Geographic Crime Heatmap":

    st.subheader("Geographic Crime Hotspots (DBSCAN)")

    df_geo = df[['Latitude', 'Longitude']].dropna()

    # Limit rows for Cloud safety
    df_geo = df_geo.sample(min(5000, len(df_geo)), random_state=42)

    coords = df_geo.values

    dbscan = DBSCAN(eps=0.0025, min_samples=50)
    labels = dbscan.fit_predict(coords)

    df_geo["cluster"] = labels

    fig = px.scatter_mapbox(
        df_geo[df_geo["cluster"] != -1],
        lat="Latitude",
        lon="Longitude",
        color="cluster",
        zoom=10,
        height=600,
        mapbox_style="carto-positron"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 3️⃣ Temporal Pattern Analysis
# ==================================================
elif page == "Temporal Pattern Analysis":

    st.subheader("Temporal Pattern Analysis")

    temporal_features = ['Hour', 'Month', 'Is_Weekend']
    df_temp = df[temporal_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_temp)

    k = st.slider("Time Clusters", 3, 6, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_temp["time_cluster"] = kmeans.fit_predict(X_scaled)

    st.markdown("### Time Cluster Summary")
    st.dataframe(df_temp.groupby("time_cluster").mean())

    # Violent Crime Peak
    if 'primary_Homicide' in df.columns:
        violent_cols = [col for col in df.columns if "primary_" in col]
        df["violent_crime"] = df[violent_cols].sum(axis=1)

        hourly = df.groupby("Hour")["violent_crime"].sum()

        fig = px.bar(
            hourly,
            title="Crime Intensity by Hour"
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 4️⃣ Dimensionality Reduction
# ==================================================
elif page == "Dimensionality Reduction (PCA & t-SNE)":

    st.subheader("Dimensionality Reduction")

    features = [
        'Latitude',
        'Longitude',
        'Hour',
        'Month',
        'Is_Weekend'
    ]

    df_pca = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)

    # ---------------- PCA ----------------
    st.markdown("### PCA")

    pca = PCA()
    pca.fit(X_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(
        x=range(1, len(cumulative)+1),
        y=cumulative,
        markers=True,
        title="Cumulative Explained Variance"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2D PCA projection
    pca_2 = PCA(n_components=2)
    X_2d = pca_2.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_2d[:,0],
        "PC2": X_2d[:,1]
    })

    fig2 = px.scatter(
        pca_df.sample(min(5000, len(pca_df))),
        x="PC1",
        y="PC2",
        opacity=0.5,
        title="2D PCA Projection"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- t-SNE (Cloud Safe) ----------------
    st.markdown("### t-SNE (Sampled for Performance)")

    sample = pca_df.sample(min(3000, len(pca_df)), random_state=42)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    X_tsne = tsne.fit_transform(sample.values)

    tsne_df = pd.DataFrame({
        "TSNE1": X_tsne[:,0],
        "TSNE2": X_tsne[:,1]
    })

    fig3 = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        opacity=0.6,
        title="t-SNE Projection"
    )

    st.plotly_chart(fig3, use_container_width=True)
