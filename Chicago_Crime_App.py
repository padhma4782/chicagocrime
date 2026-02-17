
# app.py
# Production-ready Multi-page Streamlit App for PatrolIQ

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Chicago Crime Analytics",
    layout="wide"
)

st.title("🚓 Chicago Crime Analytics Dashboard")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PatrolIQ_cleaned.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    return df

df = load_data()

lat_min, lat_max = 41.6, 42.1
lon_min, lon_max = -88.0, -87.5

df_clean = df[
    df['Latitude'].between(lat_min, lat_max) &
    df['Longitude'].between(lon_min, lon_max)
].copy()
#st.write(df_clean.columns)
# --------------------------------------------------
# Load KMeans Bundle (Model + Scaler + Features)
# --------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("crime_zone_kmeans_bundle.pkl")

bundle = load_bundle()
kmeans = bundle["model"]
scaler = bundle["scaler"]
features = bundle.get('features', ['Latitude', 'Longitude'])

# Predict clusters
if "crime_zone" not in df_clean.columns:
    X = df_clean[features].dropna()
    X_scaled = scaler.transform(X)
    df_clean.loc[X.index, "crime_zone"] = kmeans.predict(X_scaled)

labels = df_clean['crime_zone'].values
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(
    centroids,
    columns=features
)[['Latitude', 'Longitude']]

#st.write(df_clean.columns)
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

# --------------------------------------------------
# 1️⃣ Crime Zones
# --------------------------------------------------
if page == "Crime Zones (K-Means)":
    st.subheader("Crime Zones (K-Means Clustering)")
    crime_gdf = gpd.GeoDataFrame(
        df_clean,
        geometry=gpd.points_from_xy(
            df_clean['Longitude'],
            df_clean['Latitude']
        ),
        crs="EPSG:4326"
    )

    # Centroid points
    centroid_gdf = gpd.GeoDataFrame(
        centroids_df,
        geometry=gpd.points_from_xy(
            centroids_df['Longitude'],
            centroids_df['Latitude']
        ),
        crs="EPSG:4326"
    )

    #Project to meters
    crime_gdf = crime_gdf.to_crs(epsg=32616)
    centroid_gdf = centroid_gdf.to_crs(epsg=32616)

    #Create circular patrol zones
    patrol_radius_m = 1000  # 1 km
    patrol_zones = centroid_gdf.buffer(patrol_radius_m)


    fig, ax = plt.subplots(figsize=(9, 9))

    # Crime points
    crime_gdf.plot(
        ax=ax,
        column='crime_zone',
        cmap='tab10',
        markersize=3,
        alpha=0.3,
        legend=False
    )

    # Patrol circles
    gpd.GeoSeries(patrol_zones).plot(
        ax=ax,
        facecolor='none',
        edgecolor='red',
        linewidth=2
    )

    # Centroids
    centroid_gdf.plot(
        ax=ax,
        color='black',
        marker='X',
        markersize=150
    )

    ax.set_title("Chicago Circular Crime Hotspot Patrol Zones (K-Means, k=8)")
    ax.set_axis_off()
    st.pyplot(fig)


    

# --------------------------------------------------
# 2️⃣ Geographic Crime Heatmap
# --------------------------------------------------
elif page == "Geographic Crime Heatmap":

    st.subheader("Geographic Crime Heatmap (DBSCAN)")

    df_geo = df_clean.dropna(subset=["Latitude", "Longitude"]).copy()
    df_geo = df_geo.sample(min(5000, len(df_geo)), random_state=42)
    coords = df_geo[["Latitude", "Longitude"]].values

    dbscan = DBSCAN(eps=0.0035, min_samples=30)
    labels = dbscan.fit_predict(coords)

    df_geo["dbscan_zone"] = labels

    fig = px.scatter_mapbox(
        df_geo[df_geo["dbscan_zone"] != -1],
        lat="Latitude",
        lon="Longitude",
        color="dbscan_zone",
        zoom=10,
        height=600,
        mapbox_style="carto-positron"
    )

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# 3️⃣ Temporal Pattern Analysis
# --------------------------------------------------
elif page == "Temporal Pattern Analysis":
    st.subheader("Temporal Pattern Analysis")
    temporal_features = ['Hour', 'Month', 'Is_Weekend']

    # -----------------------------
    # 1️⃣ Time-based Clustering
    # -----------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X_time = df_clean[temporal_features].dropna()

    scaler_time = StandardScaler()
    X_time_scaled = scaler_time.fit_transform(X_time)

    k = st.slider("Select number of time clusters", 3, 6, 4)

    kmeans_time = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    df_clean.loc[X_time.index, 'time_cluster'] = kmeans_time.fit_predict(X_time_scaled)

    st.markdown("### Time Cluster Summary")

    time_cluster_summary = (
        df_clean
        .groupby('time_cluster')[temporal_features]
        .mean()
        .round(2)
    )

    st.dataframe(time_cluster_summary)

    # -----------------------------
    # 2️⃣ Violent Crime Analysis
    # -----------------------------
    st.markdown("### 🔥 Violent Crime Peak Hours")

    violent_cols = [
        'primary_Homicide',
        'primary_Battery',
        'primary_Assault',
        'primary_Robbery',
        'primary_Sex Offense'
    ]

    df_clean['violent_crime'] = df_clean[violent_cols].sum(axis=1)

    hourly_violent = (
        df_clean
        .groupby('Hour')['violent_crime']
        .sum()
        .sort_values(ascending=False)
    )

    st.write("Top 5 Violent Crime Hours:")
    st.write(hourly_violent.head(5))

    # Bar chart
    fig_hour = px.bar(
        hourly_violent.sort_index(),
        title="Violent Crimes by Hour",
        labels={'value': 'Total Violent Crimes'}
    )

    st.plotly_chart(fig_hour, width="stretch")

    # -----------------------------
    # 3️⃣ Weekend vs Weekday
    # -----------------------------
    st.markdown("### 📅 Weekend vs Weekday Crime Comparison")

    weekend_comparison = (
        df_clean
        .groupby('Is_Weekend')
        .size()
        .rename({0: 'Weekday', 1: 'Weekend'})
    )

    fig_weekend = px.pie(
        values=weekend_comparison.values,
        names=weekend_comparison.index,
        title="Weekend vs Weekday Crime Distribution"
    )

    st.plotly_chart(fig_weekend, width="stretch")

    # -----------------------------
    # 4️⃣ Hour vs Day Heatmap
    # -----------------------------
    st.markdown("### 🌡 Crime Frequency Heatmap (Day vs Hour)")

    hourly_heatmap = (
        df_clean
        .groupby(['Day_enc', 'Hour'])
        .size()
        .unstack(fill_value=0)
    )

    fig_heatmap = px.imshow(
        hourly_heatmap,
        color_continuous_scale='Reds',
        aspect="auto",
        labels=dict(x="Hour", y="Day of Week", color="Crime Count")
    )

    st.plotly_chart(fig_heatmap, width="stretch")
# --------------------------------------------------
# 4️⃣ Dimensionality Reduction
# --------------------------------------------------
elif page == "Dimensionality Reduction (PCA & t-SNE)":
    st.subheader("Dimensionality Reduction (PCA & t-SNE)")

    # -----------------------------
    # Filter Chicago bounds
    # -----------------------------
    df_pca = df[
        df['Latitude'].between(lat_min, lat_max) &
        df['Longitude'].between(lon_min, lon_max)
    ].copy()

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    violent_cols = [
        'primary_Homicide',
        'primary_Battery',
        'primary_Assault',
        'primary_Sex Offense',
        'primary_Criminal Sexual Assault',
        'primary_Kidnapping',
        'primary_Stalking',
        'primary_Intimidation'
    ]

    nonviolent_cols = [
        'primary_Robbery',
        'primary_Theft',
        'primary_Burglary',
        'primary_Motor Vehicle Theft',
        'primary_Criminal Damage',
        'primary_Criminal Trespass',
        'primary_Deceptive Practice',
        'primary_Narcotics',
        'primary_Other Narcotic Violation',
        'primary_Gambling',
        'primary_Prostitution',
        'primary_Public Peace Violation',
        'primary_Liquor Law Violation',
        'primary_Weapons Violation'
    ]

    df_pca['violent_crime'] = df_pca[violent_cols].sum(axis=1)
    df_pca['nonviolent_crime'] = df_pca[nonviolent_cols].sum(axis=1)

    pca_features = [
        'Latitude',
        'Longitude',
        'Hour',
        'Month',
        'Is_Weekend',
        'violent_crime',
        'nonviolent_crime'
    ]

    X = df_pca[pca_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================
    # 1️⃣ PCA
    # =============================
    st.markdown("### Principal Component Analysis (PCA)")

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Scree Plot
    fig_scree = px.line(
        x=range(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        markers=True,
        labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
        title="PCA Scree Plot"
    )
    st.plotly_chart(fig_scree, width="stretch")

    st.write(
        f"Variance explained by first 3 components: "
        f"{round(cumulative_variance[2] * 100, 2)}%"
    )

    # 2D PCA projection
    pca_2 = PCA(n_components=2)
    X_pca_2d = pca_2.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca_2d[:, 0],
        "PC2": X_pca_2d[:, 1],
        "Violent": df_pca.loc[X.index, 'violent_crime'] > 0,
        "Hour": df_pca.loc[X.index, 'Hour']
    })

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Violent",
        opacity=0.5,
        title="2D PCA Projection",
        hover_data=["Hour"]
    )

    st.plotly_chart(fig_pca, width="stretch")
    
