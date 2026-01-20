import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
custom_css = """
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #0F1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1f2e 0%, #252d3d 100%);
    }
    
    /* Card styling for metrics */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #FF6B6B;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #1a1f2e 0%, #252d3d 100%);
        border-radius: 8px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Declaring the teams
TEAMS = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Declaring the venues
CITIES = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]


@st.cache_resource
def load_model():
    """Load the pre-trained model from pickle file."""
    model_path = Path('pipe.pkl')
    if not model_path.exists():
        st.error("❌ Model file 'pipe.pkl' not found. Please ensure it's in the correct directory.")
        st.stop()
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


pipe = load_model()

# Page title and description
st.title('🏏 IPL Win Predictor')
st.markdown(
    "Predict the win probability for IPL matches based on current match conditions.",
    unsafe_allow_html=True
)

# Create tabs
tab1, tab2 = st.tabs(["🎯 Win Predictor", "📊 IPL Stats Dashboard"])

# ==================== TAB 1: WIN PREDICTOR ====================
with tab1:
    st.markdown("---")
    st.subheader("📋 Match Details")

    # Team selection
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox(
            'Select the batting team',
            sorted(TEAMS),
            help="The team that is currently batting"
        )

    with col2:
        bowling_team = st.selectbox(
            'Select the bowling team',
            sorted(TEAMS),
            help="The team that is currently bowling"
        )

    # Validate team selection
    if batting_team == bowling_team:
        st.warning("⚠️ Please select different teams for batting and bowling!")

    # Venue selection
    city = st.selectbox(
        'Select the city where the match is being played',
        sorted(CITIES),
        help="The venue for the match"
    )

    # Match information
    st.subheader("📊 Current Match Status")

    col_target = st.columns(1)[0]
    with col_target:
        target = st.number_input(
            'Target Score',
            min_value=0,
            value=150,
            help="The target score the batting team needs to achieve"
        )

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input(
            'Current Score',
            min_value=0,
            value=0,
            help="Runs scored by the batting team so far"
        )

    with col4:
        overs = st.number_input(
            'Overs Completed',
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            help="Number of overs completed (e.g., 5.3 means 5 overs and 3 balls)"
        )

    with col5:
        wickets = st.number_input(
            'Wickets Lost',
            min_value=0,
            max_value=10,
            value=0,
            help="Number of wickets lost by the batting team"
        )


    # Input validation
    def validate_inputs(batting_team: str, bowling_team: str, overs: float, score: int, target: int, wickets: int) -> Tuple[bool, str]:
        """Validate the input parameters."""
        if batting_team == bowling_team:
            return False, "Batting and bowling teams must be different."
        if overs < 0 or overs > 20:
            return False, "Overs must be between 0 and 20."
        if wickets < 0 or wickets > 10:
            return False, "Wickets must be between 0 and 10."
        if score < 0:
            return False, "Score cannot be negative."
        if target <= 0:
            return False, "Target must be greater than 0."
        if score > target + 100:  # Reasonable sanity check
            return False, "Score seems unreasonably high compared to target."
        return True, ""


    st.markdown("---")

    if st.button('🏏 Predict Probability', use_container_width=True):
        # Validate inputs
        is_valid, error_msg = validate_inputs(batting_team, bowling_team, overs, score, target, wickets)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            try:
                # Calculate derived metrics
                runs_left = target - score
                balls_left = 120 - (overs * 6)
                wickets_remaining = 10 - wickets
                
                # Handle division by zero cases
                if overs > 0:
                    current_run_rate = score / overs
                else:
                    current_run_rate = 0
                
                if balls_left > 0:
                    required_run_rate = (runs_left * 6) / balls_left
                else:
                    required_run_rate = 0
                
                # Create input dataframe
                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets': [wickets_remaining],
                    'total_runs_x': [target],
                    'cur_run_rate': [current_run_rate],
                    'req_run_rate': [required_run_rate]
                })
                
                # Make prediction
                result = pipe.predict_proba(input_df)
                loss_prob = result[0][0]
                win_prob = result[0][1]
                
                # Display results
                st.markdown("---")
                st.subheader("🏆 Win Probability Results")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        label=batting_team,
                        value=f"{round(win_prob * 100)}%",
                        delta="Batting team",
                        delta_color="off"
                    )
                
                with col_result2:
                    st.metric(
                        label=bowling_team,
                        value=f"{round(loss_prob * 100)}%",
                        delta="Bowling team",
                        delta_color="off"
                    )
                
                # Create and display bar chart
                st.subheader("📊 Probability Comparison Chart")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                teams_list = [batting_team, bowling_team]
                probabilities = [win_prob * 100, loss_prob * 100]
                colors = ['#1f77b4', '#ff7f0e']  # Blue for batting, orange for bowling
                
                bars = ax.bar(teams_list, probabilities, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{round(prob)}%',
                        ha='center', va='bottom', fontsize=14, fontweight='bold'
                    )
                
                ax.set_ylabel('Win Probability (%)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Team', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # Show match context
                st.subheader("📈 Match Context")
                context_col1, context_col2, context_col3, context_col4 = st.columns(4)
                
                with context_col1:
                    st.metric("Runs Left", runs_left)
                with context_col2:
                    st.metric("Balls Left", int(balls_left))
                with context_col3:
                    st.metric("Current Run Rate", f"{current_run_rate:.2f}")
                with context_col4:
                    st.metric("Required Run Rate", f"{required_run_rate:.2f}")
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your inputs and try again.")


# ==================== TAB 2: IPL STATS DASHBOARD ====================
with tab2:
    st.subheader("📊 IPL Historical Statistics")
    
    # Load matches data
    matches_df = pd.read_csv('matches.csv')
    
    # Remove rows with missing winner
    matches_df = matches_df[matches_df['winner'].notna()]
    
    # Create three columns for stats
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric("Total Matches Played", len(matches_df))
    
    with stat_col2:
        st.metric("Total Teams", matches_df['team1'].nunique())
    
    with stat_col3:
        st.metric("Seasons", matches_df['Season'].nunique())
    
    st.markdown("---")
    
    # 1. Top Teams by Win Percentage
    st.subheader("🏆 Top Teams by Win Percentage")
    
    team_stats = []
    all_teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
    
    for team in all_teams:
        team_matches = len(matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)])
        wins = len(matches_df[matches_df['winner'] == team])
        win_pct = (wins / team_matches * 100) if team_matches > 0 else 0
        team_stats.append({'Team': team, 'Matches': team_matches, 'Wins': wins, 'Win %': win_pct})
    
    team_stats_df = pd.DataFrame(team_stats).sort_values('Win %', ascending=False)
    
    fig_teams, ax_teams = plt.subplots(figsize=(8, 5))
    top_teams = team_stats_df.head(8)
    ax_teams.barh(top_teams['Team'], top_teams['Win %'], color='#2E86AB', edgecolor='black', alpha=0.8)
    ax_teams.set_xlabel('Win Percentage (%)', fontweight='bold')
    ax_teams.set_title('Top 8 Teams by Win %', fontweight='bold', fontsize=12)
    ax_teams.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_teams, use_container_width=True)
    
    st.dataframe(team_stats_df.head(10), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 2. Most Successful Chases
    st.subheader("🎯 Most Successful Chase Teams")
    
    # Create a function to identify if a chase was successful (team batting second won)
    def identify_chases(row):
        if row['toss_decision'] == 'bat':
            chasing_team = row['team2']
        else:
            chasing_team = row['team1']
        return chasing_team == row['winner']
    
    matches_df['is_chase'] = matches_df.apply(identify_chases, axis=1)
    
    chase_stats = []
    for team in all_teams:
        chase_attempts = len(matches_df[((matches_df['team1'] == team) & (matches_df['toss_decision'] == 'bat')) |
                                        ((matches_df['team2'] == team) & (matches_df['toss_decision'] == 'field'))])
        successful_chases = len(matches_df[((matches_df['team1'] == team) & (matches_df['toss_decision'] == 'bat') & (matches_df['winner'] == team)) |
                                           ((matches_df['team2'] == team) & (matches_df['toss_decision'] == 'field') & (matches_df['winner'] == team))])
        chase_pct = (successful_chases / chase_attempts * 100) if chase_attempts > 0 else 0
        
        if chase_attempts > 5:  # Only include teams with significant chase attempts
            chase_stats.append({'Team': team, 'Chase Attempts': chase_attempts, 'Successful': successful_chases, 'Chase %': chase_pct})
    
    chase_stats_df = pd.DataFrame(chase_stats).sort_values('Chase %', ascending=False)
    
    if len(chase_stats_df) > 0:
        fig_chase, ax_chase = plt.subplots(figsize=(8, 5))
        top_chasers = chase_stats_df.head(8)
        ax_chase.barh(top_chasers['Team'], top_chasers['Chase %'], color='#A23B72', edgecolor='black', alpha=0.8)
        ax_chase.set_xlabel('Chase Success Rate (%)', fontweight='bold')
        ax_chase.set_title('Teams with Best Chase Records', fontweight='bold', fontsize=12)
        ax_chase.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_chase, use_container_width=True)
        
        st.dataframe(chase_stats_df.head(10), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 3. City-wise Win Bias
    st.subheader("🌆 City-wise Win Distribution")
    
    city_stats = []
    for city in matches_df['city'].unique():
        if pd.notna(city):
            city_matches = matches_df[matches_df['city'] == city]
            total = len(city_matches)
            city_stats.append({'City': city, 'Matches': total})
    
    city_stats_df = pd.DataFrame(city_stats).sort_values('Matches', ascending=False)
    
    fig_city, ax_city = plt.subplots(figsize=(10, 6))
    top_cities = city_stats_df.head(10)
    ax_city.bar(range(len(top_cities)), top_cities['Matches'], color='#F18F01', edgecolor='black', alpha=0.8)
    ax_city.set_xticks(range(len(top_cities)))
    ax_city.set_xticklabels(top_cities['City'], rotation=45, ha='right')
    ax_city.set_ylabel('Number of Matches', fontweight='bold')
    ax_city.set_title('Top 10 Cities with Most IPL Matches', fontweight='bold', fontsize=12)
    ax_city.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_city, use_container_width=True)
    
    st.dataframe(city_stats_df.head(15), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 4. Season-wise Analysis
    st.subheader("📅 Season-wise Match Distribution")
    
    season_stats = matches_df['Season'].value_counts().sort_index()
    
    fig_season, ax_season = plt.subplots(figsize=(12, 5))
    ax_season.plot(season_stats.index, season_stats.values, marker='o', linewidth=2.5, markersize=8, color='#06A77D')
    ax_season.fill_between(range(len(season_stats)), season_stats.values, alpha=0.3, color='#06A77D')
    ax_season.set_xlabel('Season', fontweight='bold')
    ax_season.set_ylabel('Number of Matches', fontweight='bold')
    ax_season.set_title('IPL Matches Per Season', fontweight='bold', fontsize=12)
    ax_season.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_season, use_container_width=True)