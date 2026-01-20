import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🏏 IPL AI Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ================= TEAM COLORS =================
team_colors = {
    'Mumbai Indians': ['#004BA0','#FFC72C'],
    'Chennai Super Kings': ['#FEE101','#003C71'],
    'Royal Challengers Bangalore': ['#DA1818','#000000'],
    'Kolkata Knight Riders': ['#3B0A45','#FFD700'],
    'Delhi Capitals': ['#005DAA','#FF2B2B'],
    'Rajasthan Royals': ['#FAB5E3','#1A237E'],
    'Sunrisers Hyderabad': ['#FF671F','#FA4616'],
    'Kings XI Punjab': ['#E42313','#FFD700']
}

# ================= DATA =================
teams = list(team_colors.keys())
cities = ['Hyderabad','Bangalore','Mumbai','Kolkata','Delhi','Chennai','Jaipur',
          'Ahmedabad','Pune','Dubai','Sharjah','Abu Dhabi']

players = {
    "Mumbai Indians": ["Rohit Sharma","Surya Yadav","Hardik Pandya"],
    "Chennai Super Kings": ["MS Dhoni","Ruturaj Gaikwad","Jadeja"],
    "Royal Challengers Bangalore": ["Virat Kohli","Faf du Plessis","Maxwell"],
    "Kolkata Knight Riders": ["Shreyas Iyer","Russell","Narine"],
    "Delhi Capitals": ["Warner","Pant","Axar Patel"],
    "Rajasthan Royals": ["Sanju Samson","Jos Buttler","Boult"],
    "Sunrisers Hyderabad": ["Klaasen","Markram","Bhuvneshwar"],
    "Kings XI Punjab": ["Shikhar Dhawan","Livingstone","Rabada"]
}

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;color:#f9d423'>🏏 IPL AI Dashboard</h1>", unsafe_allow_html=True)

# ================= SIDEBAR INPUTS =================
st.sidebar.title("🏏 Match Inputs")
batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox("Bowling Team", [team for team in teams if team != batting_team])
striker = st.sidebar.selectbox("Striker", players[batting_team])
non_striker = st.sidebar.selectbox("Non-Striker", [p for p in players[batting_team] if p != striker])
bowler = st.sidebar.selectbox("Current Bowler", players[bowling_team])
city = st.sidebar.selectbox("Match City", cities)

st.sidebar.markdown("## ⚡ Match Situation")
target = st.sidebar.number_input("Target", 0)
score = st.sidebar.number_input("Current Score", 0)
overs = st.sidebar.number_input("Overs Completed", 0.0, step=0.1)
wickets_fallen = st.sidebar.number_input("Wickets Fallen", 0, 10)

# ================= LOAD MODEL =================
pipe = pickle.load(open("pipe.pkl","rb"))

# ================= PREDICTION =================
if st.sidebar.button("🚀 Predict Match Outcome"):

    # Calculation
    runs_left = target - score
    balls_left = max(1, 120 - int(overs*6))
    wickets_left = 10 - wickets_fallen
    cur_rr = score / overs if overs>0 else 0
    req_rr = (runs_left*6)/balls_left if balls_left>0 else 0

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets_left],
        'total_runs_x':[target],
        'cur_run_rate':[cur_rr],
        'req_run_rate':[req_rr]
    })

    result = pipe.predict_proba(input_df)
    win_prob = result[0][1]*100

    # ================= CELEBRATION =================
    if win_prob > 85:
        st.balloons()  # confetti balloons

    team_color = team_colors[batting_team][0]

    # ================= WIN PROB GAUGE =================
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_prob,
        delta={'reference':50},
        title={'text': f"{batting_team} Win Probability (%)"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': team_color},
            'steps':[{'range':[0,40],'color':'#ff4b4b'},
                     {'range':[40,70],'color':'#ffa500'},
                     {'range':[70,100],'color':'#00ff99'}]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ================= RUN RATE TREND =================
    overs_completed = list(range(1, int(overs)+1))
    current_rrs = [cur_rr for _ in overs_completed]
    req_rrs = [req_rr for _ in overs_completed]
    df_rr = pd.DataFrame({"Over":overs_completed, "Current RR":current_rrs, "Required RR":req_rrs})
    fig_rr = px.line(df_rr, x="Over", y=["Current RR","Required RR"],
                     title="Run Rate Trend", markers=True,
                     color_discrete_sequence=[team_colors[batting_team][0], team_colors[bowling_team][0]])
    st.plotly_chart(fig_rr, use_container_width=True)

    # ================= PLAYER IMPACT RADAR =================
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[win_prob*0.6, 100-win_prob*0.5, win_prob*0.7],
        theta=[striker, non_striker, bowler],
        fill='toself',
        name='Impact Score',
        marker_color=team_color
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                            showlegend=False, title="Player Impact Radar")
    st.plotly_chart(fig_radar, use_container_width=True)

    # ================= SCORE PROGRESSION =================
    cumulative_scores = list(range(0, score+1, max(1, score//max(1,int(overs)))))
    cumulative_target = [target/len(cumulative_scores)*i for i in range(1,len(cumulative_scores)+1)]
    df_score = pd.DataFrame({"Over":range(1,len(cumulative_scores)+1),
                             "Score":cumulative_scores,
                             "Target":cumulative_target})
    fig_area = px.area(df_score, x="Over", y=["Score","Target"], title="Score Progression",
                       color_discrete_sequence=[team_color, team_colors[bowling_team][0]])
    st.plotly_chart(fig_area, use_container_width=True)

    # ================= AI COMMENTARY =================
    st.markdown("## 🎙 AI Live Commentary")
    if win_prob > 75:
        st.success(f"🏆 {batting_team} is cruising towards victory!")
    elif win_prob > 45:
        st.warning("⚖️ Match is finely balanced. Every ball matters!")
    else:
        st.error(f"🔥 {bowling_team} is dominating the game!")

    # ================= MATCH BADGES =================
    st.markdown("## 🏅 Match Badges")
    if win_prob > 80: st.success("🔥 DOMINATOR")
    if wickets_left >= 7: st.success("🧱 STRONG BATTING DEPTH")
    if req_rr < cur_rr: st.success("⚡ CHASE UNDER CONTROL")

    # ================= AI EXPLANATION =================
    with st.expander("🤖 How AI Predicts This"):
        st.write("""
        • Team strength  
        • Venue advantage  
        • Runs & balls remaining  
        • Wickets in hand  
        • Run rate pressure  

        Model: Logistic Regression (Probability based)
        """)
    
    st.success("✅ Prediction Complete! 🏏")
