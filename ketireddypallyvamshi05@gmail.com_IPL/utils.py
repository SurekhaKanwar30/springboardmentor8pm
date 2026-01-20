import pandas as pd
import numpy as np


# ======================================================
# TEAM STRENGTH (FROM matches.csv ONLY)
# ======================================================
def compute_team_strength(matches: pd.DataFrame) -> dict:
    """
    Computes historical win-rate for each IPL team.
    Uses only matches.csv (no leakage from deliveries).
    """
    team_matches = {}
    team_wins = {}

    for _, row in matches.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]

        team_matches[t1] = team_matches.get(t1, 0) + 1
        team_matches[t2] = team_matches.get(t2, 0) + 1

        team_wins[winner] = team_wins.get(winner, 0) + 1

    team_strength = {
        team: team_wins.get(team, 0) / team_matches[team]
        for team in team_matches
    }

    return team_strength


# ======================================================
# VENUE CHASE BIAS
# ======================================================
def compute_venue_chase_bias(matches: pd.DataFrame) -> dict:
    """
    Probability of chasing team winning at each venue.
    """
    venue_bias = (
        matches.assign(chase_win=matches["winner"] == matches["team2"])
        .groupby("venue")["chase_win"]
        .mean()
        .to_dict()
    )

    return venue_bias


# ======================================================
# MATCH PHASE
# ======================================================
def get_match_phase(over: float) -> str:
    """
    Returns match phase based on over number.
    """
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    else:
        return "death"


# ======================================================
# MOMENTUM FEATURES (USED IN TRAINING)
# ======================================================
def add_momentum_features(df: pd.DataFrame, windows=(6, 12, 18)) -> pd.DataFrame:
    """
    Adds rolling runs and wickets features using total_runs and is_wicket.
    """
    for w in windows:
        df[f"runs_last_{w}"] = (
            df.groupby("match_id")["total_runs"]
            .rolling(w)
            .sum()
            .reset_index(0, drop=True)
        )

        df[f"wkts_last_{w}"] = (
            df.groupby("match_id")["is_wicket"]
            .rolling(w)
            .sum()
            .reset_index(0, drop=True)
        )

    return df


# ======================================================
# RATE + PRESSURE FEATURES
# ======================================================
def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes CRR, RRR and pressure safely.
    """
    df["current_run_rate"] = (
        df["current_score"] * 6 / df["ball_number"].clip(lower=1)
    )

    df["required_run_rate"] = (
        df["runs_remaining"] * 6 / df["balls_remaining"].clip(lower=1)
    )

    df["pressure"] = df["required_run_rate"] - df["current_run_rate"]

    return df


# ======================================================
# STREAMLIT INPUT PREPARATION (CRITICAL)
# ======================================================
def prepare_streamlit_input(
    batting_team: str,
    bowling_team: str,
    venue: str,
    over: float,
    current_score: int,
    wickets_fallen: int,
    target: int,
    team_strength: dict,
    venue_bias: dict,
    runs_last_6: int = 0,
    runs_last_12: int = 0,
    runs_last_18: int = 0,
    wkts_last_6: int = 0,
    wkts_last_12: int = 0,
    wkts_last_18: int = 0,
) -> pd.DataFrame:
    """
    Builds a single-row dataframe EXACTLY matching model features.
    Used by Streamlit app.
    """

    ball_number = int(over * 6)
    balls_remaining = max(120 - ball_number, 1)
    runs_remaining = target - current_score

    crr = (current_score * 6) / max(ball_number, 1)
    rrr = (runs_remaining * 6) / balls_remaining

    data = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "phase": get_match_phase(over),
        "current_score": current_score,
        "balls_remaining": balls_remaining,
        "wickets_remaining": 10 - wickets_fallen,
        "runs_remaining": runs_remaining,
        "current_run_rate": crr,
        "required_run_rate": rrr,
        "pressure": rrr - crr,
        "strength_diff": (
            team_strength.get(batting_team, 0.5)
            - team_strength.get(bowling_team, 0.5)
        ),
        "venue_chase_bias": venue_bias.get(venue, 0.5),
        "runs_last_6": runs_last_6,
        "runs_last_12": runs_last_12,
        "runs_last_18": runs_last_18,
        "wkts_last_6": wkts_last_6,
        "wkts_last_12": wkts_last_12,
        "wkts_last_18": wkts_last_18,
    }

    return pd.DataFrame([data])


# ======================================================
# DATA SANITY CHECK (SAFETY)
# ======================================================
def sanity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes impossible cricket states.
    """
    df = df[df["balls_remaining"] >= 0]
    df = df[df["wickets_remaining"].between(0, 10)]
    df = df[df["runs_remaining"] >= -10]
    return df
