import streamlit as st
import pandas as pd
import joblib
import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from optimizer import optimize_campaign


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CampaignIQ",
    page_icon="✦",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_conversion_model.pkl"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F7F3EE;
    color: #291C29;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Main headings */

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4.2rem;
    line-height: 1.05;
    color: #291C29;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #6E6268;
    max-width: 650px;
    line-height: 1.7;
}

/* Brand */

.brand {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #291C29;
    margin-bottom: 3rem;
}

.brand span {
    color: #E66A2C;
}

/* Cards */

.card {
    background: #FFFDFC;
    border: 1px solid #E8DED5;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #291C29;
    margin-bottom: 8px;
}

.card-text {
    color: #756A70;
    line-height: 1.6;
}

/* Metrics */

.metric-card {
    background: #291C29;
    border-radius: 18px;
    padding: 22px;
    color: white;
}

.metric-label {
    font-size: 0.85rem;
    color: #D9CDD4;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 5px;
}

/* Section headings */

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #291C29;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}

/* Buttons */

.stButton > button {
    background: #E66A2C;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: #D85D21;
    color: white;
}

/* Inputs */

div[data-baseweb="select"] > div,
input {
    border-radius: 10px !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #8B7F84;
    padding: 40px 0 10px 0;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="brand">Campaign<span>IQ</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Make every campaign<br>decision count.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
    CampaignIQ uses machine learning to predict campaign conversion
    performance and identify promising campaign configurations before launch.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------
# INTRO CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">01 · Predict</div>
        <div class="card-text">
        Estimate the expected conversion rate from campaign details.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">02 · Compare</div>
        <div class="card-text">
        Evaluate different machine learning models using standard metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">03 · Optimize</div>
        <div class="card-text">
        Explore campaign combinations and identify the highest predicted
        conversion rate.
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# CAMPAIGN BUILDER
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Build your campaign</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Enter campaign details to generate a machine-learning prediction.",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    budget = st.number_input(
        "Budget",
        min_value=0.0,
        value=6000.0,
        step=500.0
    )

    duration = st.number_input(
        "Duration (days)",
        min_value=1,
        value=15,
        step=1
    )

    platform = st.selectbox(
        "Platform",
        ["Instagram", "LinkedIn", "Facebook", "YouTube", "Google"]
    )

with col2:
    content_type = st.selectbox(
        "Content Type",
        ["Carousel", "Text", "Video", "Image"]
    )

    target_age = st.selectbox(
        "Target Age",
        ["18-24", "25-34", "35-44", "45-54", "55+"]
    )

    target_gender = st.selectbox(
        "Target Gender",
        ["Male", "Female", "Other"]
    )

with col3:
    region = st.selectbox(
        "Region",
        ["Asia", "Africa", "Europe", "North America", "South America"]
    )

    month = st.selectbox(
        "Campaign Month",
        list(range(1, 13)),
        index=8
    )

    st.markdown("<br>", unsafe_allow_html=True)

    run_analysis = st.button(
        "Analyse Campaign →",
        width="stretch"
    )


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if run_analysis:

    # ----------------------------------------------
    # PREPARE INPUT
    # ----------------------------------------------

    quarter = (month - 1) // 3 + 1

    campaign = pd.DataFrame([{
        "Budget": budget,
        "Duration": duration,
        "Platform": platform,
        "Content_Type": content_type,
        "Target_Age": target_age,
        "Target_Gender": target_gender,
        "Region": region,
        "Month": month,
        "Quarter": quarter
    }])

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    prediction = model.predict(campaign)[0]

    prediction_percent = prediction * 100

    # ----------------------------------------------
    # OPTIMIZATION
    # ----------------------------------------------

    best_config, results = optimize_campaign(
        budget=budget,
        duration=duration,
        month=month
    )

    optimized_prediction = best_config[
        "Predicted Conversion Rate"
    ]

    optimized_percent = optimized_prediction * 100

    # ----------------------------------------------
    # RESULT HEADER
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Campaign analysis</div>',
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Conversion Rate</div>
                <div class="metric-value">{prediction_percent:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Optimized Estimate</div>
                <div class="metric-value">{optimized_percent:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        improvement = optimized_percent - prediction_percent

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Difference</div>
                <div class="metric-value">{improvement:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------------------------
    # SELECTED CAMPAIGN
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Selected campaign</div>',
        unsafe_allow_html=True
    )

    selected_col1, selected_col2 = st.columns(2)

    with selected_col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Campaign details</div>
        """, unsafe_allow_html=True)

        st.write(f"**Budget:** {budget:,.0f}")
        st.write(f"**Duration:** {duration} days")
        st.write(f"**Platform:** {platform}")
        st.write(f"**Content Type:** {content_type}")

        st.markdown("</div>", unsafe_allow_html=True)

    with selected_col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Target audience</div>
        """, unsafe_allow_html=True)

        st.write(f"**Age:** {target_age}")
        st.write(f"**Gender:** {target_gender}")
        st.write(f"**Region:** {region}")
        st.write(f"**Month:** {month}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------
    # OPTIMIZATION RESULT
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Optimization recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
        <div class="card-title">
        Highest predicted campaign configuration
        </div>
        <div class="card-text">
        CampaignIQ evaluated multiple possible combinations of platform,
        content type and audience characteristics using the trained model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    opt_col1, opt_col2, opt_col3 = st.columns(3)

    with opt_col1:
        st.write(f"**Platform**")
        st.write(best_config["Platform"])

        st.write(f"**Content Type**")
        st.write(best_config["Content Type"])

    with opt_col2:
        st.write(f"**Target Age**")
        st.write(best_config["Target Age"])

        st.write(f"**Target Gender**")
        st.write(best_config["Target Gender"])

    with opt_col3:
        st.write(f"**Region**")
        st.write(best_config["Region"])

        st.write(f"**Predicted Conversion Rate**")
        st.write(f"### {optimized_percent:.2f}%")

    st.caption(
        f"Model estimate based on {len(results):,} possible campaign configurations."
    )

    # ----------------------------------------------
    # COMPARISON
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Current vs optimized</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Campaign": ["Your Campaign", "Optimized Configuration"],
        "Predicted Conversion Rate": [
            prediction_percent,
            optimized_percent
        ]
    })

    st.bar_chart(
        comparison.set_index("Campaign"),
        width="stretch"
    )

    # ----------------------------------------------
    # TOP CONFIGURATIONS
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Top campaign configurations</div>',
        unsafe_allow_html=True
    )

    top_results = results.head(10).copy()

    top_results["Predicted Conversion Rate"] = (
        top_results["Predicted Conversion Rate"] * 100
    )

    top_results = top_results.rename(
        columns={
            "Content_Type": "Content Type",
            "Target_Age": "Target Age",
            "Target_Gender": "Target Gender",
            "Predicted Conversion Rate":
                "Predicted Conversion Rate (%)"
        }
    )

    st.dataframe(
        top_results[
            [
                "Platform",
                "Content Type",
                "Target Age",
                "Target Gender",
                "Region",
                "Predicted Conversion Rate (%)"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    # ----------------------------------------------
    # KEY INSIGHTS
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Key insights</div>',
        unsafe_allow_html=True
    )

    insight1, insight2 = st.columns(2)

    with insight1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Prediction</div>
            <div class="card-text">
            The model estimates campaign conversion performance from
            pre-launch campaign characteristics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with insight2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Optimization</div>
            <div class="card-text">
            The optimizer systematically evaluates possible campaign
            configurations and identifies the highest model-predicted result.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------
    # MODEL PERFORMANCE
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True
    )

    model_results = pd.DataFrame({
        "Model": [
            "Ridge Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE": [
            0.2517,
            0.2479,
            0.2491
        ],
        "RMSE": [
            0.2906,
            0.2858,
            0.2881
        ],
        "R²": [
            -0.0064,
            0.0270,
            0.0107
        ]
    })

    st.dataframe(
        model_results,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Random Forest was selected as the final model based on the evaluation results."
    )

    st.warning(
        "Predictions are model estimates based on the available dataset and "
        "should not be treated as guaranteed campaign outcomes."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    CampaignIQ · Marketing Campaign Impact Prediction & Optimization
    <br>
    Machine Learning Project
</div>
""", unsafe_allow_html=True)