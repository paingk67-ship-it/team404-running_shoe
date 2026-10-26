"""Prompt B: Team 404 — Data-Driven Value Assessment of Running Shoes
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Team 404 | Running Shoe Value", page_icon="👟", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------- Data ----------
st.title("👟 Data-Driven Value Assessment of Running Shoes")
st.caption("Team 404 · Price vs. performance analytics")

upload = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")
try:
    df = load_data(upload if upload else "running_shoes.csv")
except FileNotFoundError:
    st.error("running_shoes.csv not found. Run `python generate_data.py` or upload a CSV.")
    st.stop()

required = {"brand", "price_usd", "weight_g", "cushioning_score", "energy_return_pct", "grip_index"}
if not required.issubset(df.columns):
    st.error(f"CSV is missing columns: {sorted(required - set(df.columns))}")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
p_min, p_max = float(df.price_usd.min()), float(df.price_usd.max())
w_min, w_max = float(df.weight_g.min()), float(df.weight_g.max())
price_rng = st.sidebar.slider("Price (USD)", p_min, p_max, (p_min, p_max))
weight_rng = st.sidebar.slider("Weight (g)", w_min, w_max, (w_min, w_max))
brands = st.sidebar.multiselect("Brand", sorted(df.brand.unique()), default=sorted(df.brand.unique()))

fdf = df[df.price_usd.between(*price_rng) & df.weight_g.between(*weight_rng) & df.brand.isin(brands)].copy()
if len(fdf) < 10:
    st.warning("Too few shoes match the filters (need at least 10). Widen the filters.")
    st.stop()

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shoes", len(fdf))
    c2.metric("Avg price", f"${fdf.price_usd.mean():.0f}")
    c3.metric("Avg energy return", f"{fdf.energy_return_pct.mean():.1f}%")
    c4.metric("Avg weight", f"{fdf.weight_g.mean():.0f} g")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Price vs Performance", "🧩 Clustering", "🌲 Feature Importance", "🗂 Data"])

# ---------- Tab 1 ----------
with tab1:
    y_choice = st.radio("Y-axis", ["energy_return_pct", "cushioning_score"], horizontal=True)
    fig = px.scatter(fdf, x="price_usd", y=y_choice, color="brand", size="weight_g",
                     hover_data=["model_name", "weight_g"], trendline="ols",
                     trendline_scope="overall", trendline_color_override="black",
                     title=f"Price vs {y_choice.replace('_', ' ').title()}")
    fig.update_layout(template="plotly_white", height=550)
    st.plotly_chart(fig, width="stretch")

# ---------- Tab 2 ----------
CL_FEATS = ["price_usd", "cushioning_score", "energy_return_pct", "weight_g"]
with tab2:
    k = st.slider("Number of clusters (K)", 2, 5, 3)
    X = StandardScaler().fit_transform(fdf[CL_FEATS])
    fdf["cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(X).astype(str)
    pca = PCA(2, random_state=42)
    pcs = pca.fit_transform(X)
    fdf["PC1"], fdf["PC2"] = pcs[:, 0], pcs[:, 1]
    left, right = st.columns([3, 2])
    with left:
        fig = px.scatter(fdf, x="PC1", y="PC2", color="cluster", hover_data=["brand", "model_name"] + CL_FEATS,
                         title="Clusters (PCA projection)",
                         labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
                                 "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"})
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Cluster summary")
        summ = fdf.groupby("cluster")[CL_FEATS].mean().round(2)
        summ.insert(0, "n_shoes", fdf.cluster.value_counts().sort_index())
        st.dataframe(summ, width="stretch")

# ---------- Tab 3 ----------
with tab3:
    RF_FEATS = ["price_usd", "weight_g", "cushioning_score", "grip_index"]
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(fdf[RF_FEATS], fdf.energy_return_pct)
    imp = pd.Series(rf.feature_importances_, index=RF_FEATS).sort_values()
    fig = px.bar(imp, orientation="h", labels={"value": "Importance", "index": "Feature"},
                 title="Feature Importance: Predicting Energy Return")
    fig.update_layout(template="plotly_white", showlegend=False, height=450)
    st.plotly_chart(fig, width="stretch")

# ---------- Tab 4 ----------
with tab4:
    st.dataframe(fdf.drop(columns=["PC1", "PC2"], errors="ignore"), width="stretch")
    st.download_button("Download filtered CSV", fdf.to_csv(index=False), "filtered_shoes.csv", "text/csv")
