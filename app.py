"""Streamlit app for a transparent Hyderabad home-price ML project."""
from __future__ import annotations

import json
import html
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "house_price_model.joblib"
METRICS_FILE = ROOT / "model_metrics.json"
MONEY = "₹"
AMENITY_LABELS = {
    "Gymnasium": "Gym",
    "SwimmingPool": "Swimming pool",
    "CarParking": "Car parking",
    "PowerBackup": "Power backup",
    "24X7Security": "24×7 security",
    "LiftAvailable": "Lift",
}

st.set_page_config(page_title="Hyderabad Home Price Studio", page_icon="⌂", layout="wide")
st.markdown("""
<style>
:root { --ink:#172b4d; --muted:#61728a; --line:#e4eaf1; --blue:#176b87; --blue2:#eaf5f8; --green:#16794b; }
html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; color:var(--ink); }
.stApp { background:#fff; }
.block-container { max-width:1180px; padding-top:2.0rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Segoe UI',Arial,sans-serif!important; letter-spacing:-.035em; color:var(--ink); }
h1 { font-size:2.35rem!important; font-weight:800!important; }
.brand { color:var(--blue); font-size:.73rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.7rem; }
.lede { color:var(--muted); font-size:1.04rem; max-width:790px; margin-top:-.35rem; line-height:1.65; }
.surface { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.2rem 1.35rem; box-shadow:0 8px 24px rgba(23,43,77,.045); }
.result { background:linear-gradient(145deg,#f1f9fb,#fff 75%); border:1px solid #cfe5ea; border-radius:18px; padding:1.55rem; min-height:250px; }
.result-label { color:var(--muted); font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }
.result-value { color:#125e77; font-family:'Segoe UI',Arial,sans-serif; font-weight:800; font-size:2.55rem; letter-spacing:-.05em; line-height:1.2; padding:.45rem 0 .35rem; }
.small-note { color:var(--muted); font-size:.86rem; line-height:1.55; }
.metric-card { border:1px solid var(--line); border-radius:13px; padding:1rem 1.1rem; height:100%; background:#fff; }
.metric-title { color:var(--muted); font-size:.8rem; font-weight:600; }
.metric-value { font-family:'Segoe UI',Arial,sans-serif; font-size:1.35rem; font-weight:800; color:var(--ink); padding-top:.25rem; }
.step { display:inline-block; width:25px; height:25px; line-height:25px; border-radius:50%; text-align:center; color:#12627a; background:#eaf5f8; font-weight:700; font-size:.8rem; margin-right:.45rem; }
.section-title { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.08rem; color:var(--ink); margin:0 0 .75rem; }
div[data-testid="stForm"] { border:0; padding:0; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:13px; padding:14px 16px; }
.stButton>button[kind="primary"], .stFormSubmitButton>button { background:#176b87; border:0; border-radius:9px; font-weight:700; min-height:2.8rem; }
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover { background:#10566c; }
div[data-testid="stAlert"] { border-radius:12px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists() or not METRICS_FILE.exists():
        return None, None
    artifact = joblib.load(MODEL_FILE)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return artifact, metrics


def rupees(value: float, compact: bool = False) -> str:
    if compact:
        if value >= 10_000_000:
            return f"₹{value / 10_000_000:.2f} Cr"
        if value >= 100_000:
            return f"₹{value / 100_000:.2f} L"
    return f"₹{value:,.0f}"


def metric_card(title: str, value: str) -> str:
    return f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>'


artifact, metrics = load_artifacts()
st.markdown('<div class="brand">Hyderabad · Machine learning project</div>', unsafe_allow_html=True)
st.title("Hyderabad Home Price Studio")
st.markdown(
    '<p class="lede">Explore a data-based property-price estimate using the home size, bedrooms, locality, resale status, and selected amenities. The estimate learns from the supplied Hyderabad dataset; it is not a live quote or a property valuation.</p>',
    unsafe_allow_html=True,
)

if artifact is None:
    st.error("The trained model has not been created yet. From this project folder, run `python train_model.py`, then restart the app.")
    st.stop()

properties: pd.DataFrame = artifact["data"]
location_counts = artifact["location_counts"]
locations = sorted([name for name, count in location_counts.items() if int(count) >= 8])
if not locations:
    st.error("The validated dataset does not have any locations with enough rows to offer a location-specific estimate.")
    st.stop()
default_location = "Kukatpally" if "Kukatpally" in locations else locations[0]
area_min, area_max = artifact["supported_area_range"]
bedroom_values = artifact["supported_bedroom_values"]
median_area = int(properties["Area"].median())

estimate_tab, explore_tab, model_tab = st.tabs(["Estimate a home", "Explore the dataset", "Model & methodology"])

with estimate_tab:
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title"><span class="step">1</span>Tell us about the home</div>', unsafe_allow_html=True)
            st.caption("Choose a locality represented by at least 8 unique listings in the supplied file.")
            with st.form("estimate_form", border=False):
                location = st.selectbox("Hyderabad locality", locations, index=locations.index(default_location), help="Start typing to search the localities included in this dataset.")
                area = st.number_input(
                    "Built-up area (sq ft)", min_value=int(area_min), max_value=int(area_max),
                    value=max(int(area_min), min(median_area, int(area_max))), step=50,
                    help=f"The app accepts the central 98% of observed areas: {area_min:,.0f} to {area_max:,.0f} sq ft.",
                )
                bedrooms = st.selectbox("Bedrooms", bedroom_values, index=bedroom_values.index(2) if 2 in bedroom_values else 0)
                resale = st.radio("Home status", ["New", "Resale"], horizontal=True, help="This field is present in the dataset.")
                with st.expander("Optional: amenities", expanded=False):
                    st.caption("These six amenities are supported by the source columns. Choose ‘Not specified’ if you do not know; it is kept separate from ‘No’.")
                    amenity_values = {}
                    for feature, label in AMENITY_LABELS.items():
                        choice = st.selectbox(label, ["Not specified", "Yes", "No"], key=f"amenity_{feature}")
                        amenity_values[feature] = choice
                submitted = st.form_submit_button("Estimate from the dataset", type="primary", use_container_width=True)
        count = int(location_counts.get(location, 0))
        st.markdown(f'<p class="small-note" style="margin:.8rem .2rem">This locality has <b>{count:,}</b> unique rows after duplicate removal. The model was trained on {artifact["dataset_rows"]:,} unique rows across {len(location_counts):,} localities.</p>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title"><span class="step">2</span>Your estimate</div>', unsafe_allow_html=True)
        if submitted:
            values = {
                "Area": float(area),
                "No. of Bedrooms": int(bedrooms),
                "Location": location,
                "Resale": resale,
                **amenity_values,
            }
            row = pd.DataFrame([values], columns=artifact["features"])
            prediction = max(0.0, float(artifact["model"].predict(row)[0]))
            locality_count = int(location_counts[location])
            st.markdown(
                f'<div class="result"><div class="result-label">Estimated property price</div><div class="result-value">{rupees(prediction, compact=True)}</div><div class="small-note">{rupees(prediction)} · for {int(area):,} sq ft, {int(bedrooms)} bedroom(s) in {html.escape(location)}</div><hr style="border:0;border-top:1px solid #dce9ed;margin:1rem 0"><div class="small-note">This is the model’s single point estimate. It is learned from the prices recorded in the dataset; it is not a guaranteed sale price and does not include a confidence interval.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(" ")
            st.info(f"On held-out rows, the model’s average absolute error was {rupees(metrics['nested_cv_mae'])}. That is a dataset-wide evaluation measure, not a personal error range for this home.")
            if int(bedrooms) >= 6:
                st.warning("Homes with this bedroom count are uncommon in the dataset; the estimate has less supporting data.")
            if locality_count < 20:
                st.warning("This locality has a relatively small number of listings. Use extra care when interpreting the estimate.")
        else:
            st.markdown(
                '<div class="result"><div class="result-label">Ready when you are</div><div class="result-value" style="font-size:1.65rem">Set the home details</div><div class="small-note">The estimate will use the trained model and the locality you choose. Your inputs stay in this browser session.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(" ")
            st.markdown(metric_card("Cross-validation MAE", rupees(metrics["nested_cv_mae"])), unsafe_allow_html=True)
            st.markdown('<p class="small-note" style="margin-top:.6rem">Average absolute error from nested 5-fold validation. Lower is better; this does not promise the same error for a specific property.</p>', unsafe_allow_html=True)

    st.markdown(" ")
    st.warning("The source file has no collection date, bathrooms, or PIN codes. This app estimates from the available columns and should not be treated as a current-market valuation.")

with explore_tab:
    st.subheader("What the supplied data says")
    st.caption("The charts summarize the included file, not today’s full Hyderabad housing market.")
    top = properties["Price"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Unique listings", f"{len(properties):,}"),
        ("Localities", f"{properties['Location'].nunique():,}"),
        ("Median dataset price", rupees(float(top.median()), compact=True)),
        ("Median area", f"{properties['Area'].median():,.0f} sq ft"),
    ]
    for col, (title, value) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(metric_card(title, value), unsafe_allow_html=True)
    chart1, chart2 = st.columns(2, gap="large")
    with chart1:
        fig = px.scatter(
            properties, x="Area", y="Price", color="No. of Bedrooms", hover_name="Location",
            labels={"Area": "Area (sq ft)", "Price": "Dataset price (₹)", "No. of Bedrooms": "Bedrooms"},
            opacity=.62, color_continuous_scale="Teal",
        )
        fig.update_layout(template="plotly_white", height=390, margin=dict(l=5,r=5,t=20,b=5), font=dict(family="Segoe UI, Arial", color="#172b4d"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Price vs area. Very high or unusual points remain visible; the model does not hide the source data’s outliers.")
    with chart2:
        locality_summary = properties.groupby("Location").agg(listings=("Price", "size"), median_price=("Price", "median")).query("listings >= 12").nlargest(12, "median_price").sort_values("median_price").reset_index()
        fig2 = px.bar(
            locality_summary, x="median_price", y="Location", orientation="h",
            labels={"median_price": "Median dataset price (₹)", "Location": "Locality"},
            color="median_price", color_continuous_scale="Teal", hover_data={"listings": True},
        )
        fig2.update_layout(template="plotly_white", height=390, margin=dict(l=5,r=5,t=20,b=5), font=dict(family="Segoe UI, Arial", color="#172b4d"), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Higher median dataset prices among localities with at least 12 unique rows. This is descriptive, not an area ranking or recommendation.")
    with st.expander("How the source data was prepared"):
        st.write(f"The uploaded file contained {metrics['source_rows']:,} rows. {metrics['exact_duplicate_rows_removed']:,} exact duplicate rows were removed; {metrics['invalid_rows_removed']:,} invalid rows were excluded. {metrics['training_rows']:,} rows remain for model training.")
        st.write("The original file contains 40 columns. The estimator uses area, bedrooms, locality, new/resale status, and six amenities. It does not use the other amenity columns, because this keeps the user form manageable.")

with model_tab:
    st.subheader("How this estimate is produced")
    st.markdown("""
**1 · Validate and prepare.** The app checks required fields, positive prices/areas, whole-number bedroom counts, valid status codes, and duplicate rows. Exact duplicates are removed. Amenity code `9` is kept as **Not specified**, not converted to No.

**2 · Learn from multiple signals.** A scikit-learn pipeline encodes locality and amenity categories, scales area and bedroom counts for the linear model, and models the skewed price target on a log scale. The training step compares regularized linear regression with an Extra Trees ensemble.

**3 · Evaluate before fitting the final model.** Nested 5-fold cross-validation selects between the two model types inside each training fold, then measures predictions on that fold’s held-out rows. The app also compares that result with a median-price baseline. This reduces model-selection optimism, though it cannot establish performance on current listings outside this file.
""")
    st.markdown("#### Validation results")
    a, b, c, d = st.columns(4)
    metric_rows = [
        ("Nested CV · MAE", rupees(metrics["nested_cv_mae"])),
        ("Nested CV · RMSE", rupees(metrics["nested_cv_rmse"])),
        ("Nested CV · R²", f"{metrics['nested_cv_r2']:.3f}"),
        ("Median baseline · MAE", rupees(metrics["median_baseline_mae"])),
    ]
    for col, (label, value) in zip((a,b,c,d), metric_rows):
        with col:
            st.markdown(metric_card(label, value), unsafe_allow_html=True)
    st.caption("MAE and RMSE are in the source price values and shown as ₹ based on the dataset’s Hyderabad, India context; the source column itself does not contain a currency code. R² can be negative. None of these is a confidence interval.")
    st.markdown("#### What the dataset supports—and what it does not")
    st.markdown("""
- **Supported:** Hyderabad locality, area in square feet, bedroom count, new/resale status, and six selected amenities.
- **Not included:** bathrooms, PIN code, property age, floor, verified transaction status, or a collection date. The model cannot use fields that are absent.
- **Time limitation:** no date is recorded in this CSV. The estimate is not a live market price.
- **Geography limitation:** location is a locality label, not a precise map coordinate. Select a locality represented in the dropdown.
- **Price meaning:** the related dataset documentation describes Hyderabad, India housing prices and area in square feet. The CSV has no explicit currency code or asking-price/transaction definition. The app displays ₹ as an informed inference from that context, not as independently verified transaction currency.
- **Use:** compare as a learning/demo estimate alongside current listings and professional advice; do not use as a formal valuation.
""")
    with st.expander("Technical details"):
        st.write(f"Selected deployable model: **{metrics['selected_model_label']}**. Model candidates were selected by 5-fold CV MAE on the full prepared data after nested outer evaluation. The final model was then fit on all {metrics['training_rows']:,} unique valid rows.")
        st.write("The final fitted preprocessing and prediction pipeline is saved with the project. Predictions from locations unseen during training are rejected by the user interface because only supported observed localities are offered.")
    st.markdown("#### Dataset source notes")
    st.markdown("The dataset matches the publicly described Kaggle ‘Housing Prices in Metropolitan Areas of India’ dataset, with a Hyderabad-only file. See [the Hyderabad project description](https://github.com/ShefaaSaied/Predicting-Housing-Prices-in-hyderabad-India) and [the published Hyderabad machine-learning example](https://dataforsolution.github.io/assets/books/Grokking%20Machine%20Learning.pdf). The supplied CSV is treated as the project’s source of truth; its exact collection date and reuse license were not included in the file.")

st.markdown("<hr style='border:0;border-top:1px solid #e4eaf1;margin:2rem 0 1rem'>", unsafe_allow_html=True)
st.caption("Hyderabad Home Price Studio · Educational machine learning project · Built from the supplied Hyderabad.csv")
