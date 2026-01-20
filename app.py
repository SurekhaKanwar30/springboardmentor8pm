import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
pipe = pickle.load(open('pipe.pkl', 'rb'))

# -------------------------------
# Teams, Players
# -------------------------------
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

batsmen = [
    'Virat Kohli', 'Rohit Sharma', 'MS Dhoni',
    'David Warner', 'KL Rahul'
]

bowlers = [
    'Jasprit Bumrah', 'Rashid Khan',
    'Bhuvneshwar Kumar', 'Yuzvendra Chahal'
]

cities = ['Mumbai', 'Chennai', 'Delhi', 'Kolkata', 'Bangalore']

# -------------------------------
# UI
# -------------------------------
st.title("IPL Player Impact Analysis")

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
city = st.selectbox("City", cities)

target = st.number_input("Target", min_value=1)
score = st.number_input("Current Score", min_value=0)
overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0)
wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10)

player_type = st.radio("Select Player Type", ["Batsman", "Bowler"])

if player_type == "Batsman":
    key_player = st.selectbox("Select Key Batsman", batsmen)
else:
    key_player = st.selectbox("Select Key Bowler", bowlers)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("Analyze Player Impact"):

    def predict_probability(score, wickets, run_rate_adjust=0, wicket_adjust=0):
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets + wicket_adjust

        cur_rr = score / overs
        req_rr = (runs_left * 6) / balls_left

        df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'cur_run_rate': [cur_rr + run_rate_adjust],
            'req_run_rate': [req_rr]
        })

        return pipe.predict_proba(df)[0][1]

    # Base probability
    base_prob = predict_probability(score, wickets_fallen)

    # Player impact simulation
    if player_type == "Batsman":
        impact_prob = predict_probability(
            score,
            wickets_fallen,
            run_rate_adjust=0.8,     # batsman improves scoring
            wicket_adjust=1          # reduces wicket risk
        )
        st.info(f"Impact of {key_player} staying at crease")

    else:
        impact_prob = predict_probability(
            score,
            wickets_fallen,
            run_rate_adjust=-0.6,    # bowler controls run rate
            wicket_adjust=-1         # increases wicket chance
        )
        st.info(f"Impact of {key_player} bowling now")

    # -------------------------------
    # Results
    # -------------------------------
    st.subheader("Win Probability Comparison")

    st.write(f"Without Player Impact: **{round(base_prob * 100, 2)}%**")
    st.write(f"With {key_player}: **{round(impact_prob * 100, 2)}%**")

    impact_change = (impact_prob - base_prob) * 100

    if impact_change > 0:
        st.success(f"Win probability increases by **{round(impact_change, 2)}%**")
    else:
        st.error(f"Win probability decreases by **{round(abs(impact_change), 2)}%**")