import streamlit as st
import pandas as pd
import numpy as np
import pickle

from utils import (
    compute_team_strength,
    compute_venue_chase_bias,
    prepare_streamlit_input
)

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Win Probability Predictor")
st.markdown(
    "A real-time, ball-by-ball win probability model built using IPL data."
)

# ------------------------------------------------------
# MODEL FILES (DO NOT CHANGE NAMES)
# ------------------------------------------------------
MODEL_FILES = {
    "XGBoost (Advanced)": "model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Linear Regression": "linear_model.pkl",
    "Random Forest": "rf_model.pkl"
}

# ------------------------------------------------------
# LOAD MODEL & DATA
# ------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# helpers
team_strength = compute_team_strength(matches)
venue_bias = compute_venue_chase_bias(matches)

teams = sorted(matches["team1"].unique())
venues = sorted(matches["venue"].dropna().unique())

# ------------------------------------------------------
# SIDEBAR – MODEL SELECTION
# ------------------------------------------------------
st.sidebar.header("🤖 Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(MODEL_FILES.keys())
)

model = load_model(MODEL_FILES[selected_model_name])

# ------------------------------------------------------
# SIDEBAR – MATCH SETUP
# ------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🏟 Match Setup")

batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox(
    "Bowling Team", [t for t in teams if t != batting_team]
)

venue = st.sidebar.selectbox("Venue", venues)

target = st.sidebar.number_input(
    "Target Score", min_value=50, max_value=300, value=180
)

# ------------------------------------------------------
# SIDEBAR – CURRENT STATE
# ------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📊 Current Match State")

over = st.sidebar.slider("Overs Completed", 0.0, 20.0, 10.0, step=0.1)
current_score = st.sidebar.number_input(
    "Current Score", min_value=0, max_value=300, value=80
)
wickets_fallen = st.sidebar.slider("Wickets Fallen", 0, 10, 3)

# ------------------------------------------------------
# SIDEBAR – MOMENTUM
# ------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔥 Recent Momentum")

runs_last_6 = st.sidebar.slider("Runs in Last 6 Balls", 0, 36, 8)
runs_last_12 = st.sidebar.slider("Runs in Last 12 Balls", 0, 72, 16)

wkts_last_6 = st.sidebar.slider("Wickets in Last 6 Balls", 0, 3, 0)
wkts_last_12 = st.sidebar.slider("Wickets in Last 12 Balls", 0, 5, 1)

# ------------------------------------------------------
# BUILD MODEL INPUT
# ------------------------------------------------------
input_df = prepare_streamlit_input(
    batting_team=batting_team,
    bowling_team=bowling_team,
    venue=venue,
    over=over,
    current_score=current_score,
    wickets_fallen=wickets_fallen,
    target=target,
    team_strength=team_strength,
    venue_bias=venue_bias,
    runs_last_6=runs_last_6,
    runs_last_12=runs_last_12,
    runs_last_18=runs_last_12,
    wkts_last_6=wkts_last_6,
    wkts_last_12=wkts_last_12,
    wkts_last_18=wkts_last_12
)

# ------------------------------------------------------
# PREDICTION (SAFE FOR ALL MODELS)
# ------------------------------------------------------
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_df)[0][1]
else:
    prob = model.predict(input_df)[0]
    prob = float(np.clip(prob, 0, 1))

prob_pct = int(prob * 100)

# ------------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------------
st.markdown("## 📈 Win Probability")

col1, col2 = st.columns([2, 1])

with col1:
    st.progress(prob_pct)

with col2:
    st.metric(
        label=f"{batting_team} Win %",
        value=f"{prob_pct}%"
    )

# ------------------------------------------------------
# CONTEXT MESSAGE
# ------------------------------------------------------
if prob_pct >= 65:
    st.success("🟢 Batting team is in a strong position")
elif prob_pct >= 40:
    st.warning("🟡 Match is evenly balanced")
else:
    st.error("🔴 Bowling team is dominating")

# ------------------------------------------------------
# WHAT-IF SIMULATION
# ------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 What-If Simulation")

sim_col1, sim_col2, sim_col3 = st.columns(3)

if sim_col1.button("➕ +12 Runs Next Over"):
    current_score += 12

if sim_col2.button("❌ Lose 1 Wicket"):
    wickets_fallen = min(wickets_fallen + 1, 10)

if sim_col3.button("⏭ Advance 1 Over"):
    over = min(over + 1, 20)

sim_input = prepare_streamlit_input(
    batting_team,
    bowling_team,
    venue,
    over,
    current_score,
    wickets_fallen,
    target,
    team_strength,
    venue_bias,
    runs_last_6,
    runs_last_12,
    runs_last_12,
    wkts_last_6,
    wkts_last_12,
    wkts_last_12
)

if hasattr(model, "predict_proba"):
    sim_prob = model.predict_proba(sim_input)[0][1]
else:
    sim_prob = model.predict(sim_input)[0]
    sim_prob = float(np.clip(sim_prob, 0, 1))

st.info(f"Simulated Win Probability: **{int(sim_prob * 100)}%**")

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.caption(
    f"Built using IPL ball-by-ball data | "
    f"Selected Model: {selected_model_name}"
)
