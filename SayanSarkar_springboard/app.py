import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="IPL Win Probability Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    font-family: "Inter", sans-serif;
}

/* Main container spacing */
.block-container {
    padding-top: 2rem;
}

/* Headings */
h1 {
    font-weight: 700;
    color: #0f172a;
}
h2, h3 {
    color: #1e293b;
    font-weight: 600;
}

/* Glass cards */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

/* KPI number */
.kpi {
    font-size: 36px;
    font-weight: 700;
    color: #2563eb;
}

/* Labels */
.label {
    color: #64748b;
    font-size: 14px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f1f5f9);
    border-right: 1px solid #e5e7eb;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border-radius: 10px;
    padding: 12px 26px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1e40af, #4338ca);
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #6366f1);
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
pipe = pickle.load(open("pipe.pkl", "rb"))

# CONSTANTS (MATCH TRAINING DATA)
teams = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

cities = [
    'Ahmedabad','Bangalore','Chandigarh','Chennai','Delhi',
    'Dharamsala','Hyderabad','Jaipur','Kolkata','Mumbai',
    'Pune','Ranchi'
]

# SIDEBAR INPUTS (LIKE FILTER PANEL)
st.sidebar.markdown("## Match Configuration")

batting_team = st.sidebar.selectbox("Batting Team", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(teams))
city = st.sidebar.selectbox("Match City", sorted(cities))

st.sidebar.markdown("### Target")
target = st.sidebar.number_input("Target Score", min_value=1, step=1)

st.sidebar.markdown("### Match Progress")
score = st.sidebar.number_input("Current Score", min_value=0, step=1)
overs = st.sidebar.number_input("Overs Completed (e.g. 10.3)", 0.1, 20.0, 0.1)
wickets_fallen = st.sidebar.number_input("Wickets Fallen", 0, 10)

predict = st.sidebar.button("Predict Probability")

# HEADER
st.markdown("## IPL Win Probability Dashboard")
st.caption("Real-time match outcome analytics powered by machine learning")

st.divider()

# PREDICTION LOGIC
if predict:

    balls_bowled = int(overs * 6)
    balls_left = max(0, 120 - balls_bowled)
    runs_left = target - score
    wickets_left = 10 - wickets_fallen

    cur_rr = 0 if balls_bowled == 0 else (score * 6) / balls_bowled
    req_rr = 0 if balls_left == 0 else (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'cur_run_rate': [cur_rr],
        'req_run_rate': [req_rr]
    })

    result = pipe.predict_proba(input_df)
    win_prob = result[0][1]
    loss_prob = result[0][0]

    # KPI CARDS (TOP ROW)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="label">{batting_team}</div>
            <div class="kpi">{round(win_prob*100)}%</div>
            <div class="label">Win Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="label">{bowling_team}</div>
            <div class="kpi">{round(loss_prob*100)}%</div>
            <div class="label">Win Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="label">Runs Left</div>
            <div class="kpi">{runs_left}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="label">Balls Left</div>
            <div class="kpi">{balls_left}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ANALYTICS SECTION
    a1, a2 = st.columns([2, 1])

    with a1:
        st.markdown("### Probability Comparison")
        fig, ax = plt.subplots()
        ax.bar(
            [batting_team, bowling_team],
            [win_prob*100, loss_prob*100]
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

    with a2:
        st.markdown("### Match Metrics")
        st.metric("Current Run Rate", round(cur_rr, 2))
        st.metric("Required Run Rate", round(req_rr, 2))
        st.metric("Wickets Remaining", wickets_left)

    st.progress(win_prob)

else:
    st.info("Use the left panel to configure match details and generate predictions.")

# FOOTER
st.divider()
st.caption("IPL Win Probability Predictor • Machine Learning Dashboard")