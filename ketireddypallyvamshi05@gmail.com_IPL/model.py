import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

from xgboost import XGBClassifier


# ======================================================
# 1. LOAD DATA (CONFIRMED COLUMNS)
# ======================================================
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Only normal results
matches = matches[matches["result"] == "normal"]


# ======================================================
# 2. CREATE WICKET FLAG (FROM player_dismissed)
# ======================================================
deliveries["is_wicket"] = deliveries["player_dismissed"].notna().astype(int)


# ======================================================
# 3. MERGE MATCH INFO INTO DELIVERIES
# ======================================================
deliveries = deliveries.merge(
    matches[["id", "venue", "team1", "team2", "winner"]],
    left_on="match_id",
    right_on="id",
    how="inner"
)


# ======================================================
# 4. BALL & SCORE STATE (USING total_runs)
# ======================================================
deliveries["ball_number"] = deliveries["over"] * 6 + deliveries["ball"]

deliveries["current_score"] = (
    deliveries.groupby("match_id")["total_runs"].cumsum()
)

deliveries["wickets_fallen"] = (
    deliveries.groupby("match_id")["is_wicket"].cumsum()
)

deliveries["balls_remaining"] = 120 - deliveries["ball_number"]
deliveries["wickets_remaining"] = 10 - deliveries["wickets_fallen"]


# ======================================================
# 5. TARGET (SAFE, NO LEAKAGE)
# ======================================================
first_innings_score = (
    deliveries[deliveries["inning"] == 1]
    .groupby("match_id")["current_score"]
    .max()
)

deliveries = deliveries.merge(
    first_innings_score.rename("first_innings_score"),
    on="match_id",
    how="left"
)

# Only second innings for win probability
deliveries = deliveries[deliveries["inning"] == 2]

deliveries["target"] = deliveries["first_innings_score"] + 1
deliveries["runs_remaining"] = deliveries["target"] - deliveries["current_score"]


# ======================================================
# 6. RUN RATES & PRESSURE (CORE SIGNAL)
# ======================================================
deliveries["current_run_rate"] = (
    deliveries["current_score"] * 6 / deliveries["ball_number"].clip(lower=1)
)

deliveries["required_run_rate"] = (
    deliveries["runs_remaining"] * 6 / deliveries["balls_remaining"].clip(lower=1)
)

deliveries["pressure"] = (
    deliveries["required_run_rate"] - deliveries["current_run_rate"]
)


# ======================================================
# 7. MOMENTUM FEATURES (ROBUST, SAFE WINDOWS)
# ======================================================
for w in [6, 12]:
    deliveries[f"runs_last_{w}"] = (
        deliveries.groupby("match_id")["total_runs"]
        .rolling(w)
        .sum()
        .reset_index(0, drop=True)
    )

    deliveries[f"wkts_last_{w}"] = (
        deliveries.groupby("match_id")["is_wicket"]
        .rolling(w)
        .sum()
        .reset_index(0, drop=True)
    )


# ======================================================
# 8. MATCH PHASE
# ======================================================
def match_phase(over):
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    else:
        return "death"

deliveries["phase"] = deliveries["over"].apply(match_phase)


# ======================================================
# 9. TEAM STRENGTH (FROM matches.csv ONLY)
# ======================================================
team_matches = {}
team_wins = {}

for _, row in matches.iterrows():
    for team in [row["team1"], row["team2"]]:
        team_matches[team] = team_matches.get(team, 0) + 1
    team_wins[row["winner"]] = team_wins.get(row["winner"], 0) + 1

team_strength = {
    team: team_wins.get(team, 0) / team_matches[team]
    for team in team_matches
}

deliveries["strength_diff"] = (
    deliveries["batting_team"].map(team_strength)
    - deliveries["bowling_team"].map(team_strength)
)


# ======================================================
# 10. VENUE CHASE BIAS
# ======================================================
venue_bias = (
    matches.assign(chase_win=matches["winner"] == matches["team2"])
    .groupby("venue")["chase_win"]
    .mean()
)

deliveries["venue_chase_bias"] = deliveries["venue"].map(venue_bias)


# ======================================================
# 11. TARGET LABEL
# ======================================================
deliveries["win"] = (
    deliveries["batting_team"] == deliveries["winner"]
).astype(int)


# ======================================================
# 12. FINAL FEATURE SET (STABLE & MEANINGFUL)
# ======================================================
FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
    "phase",
    "current_score",
    "balls_remaining",
    "wickets_remaining",
    "runs_remaining",
    "current_run_rate",
    "required_run_rate",
    "pressure",
    "strength_diff",
    "venue_chase_bias",
    "runs_last_6",
    "runs_last_12",
    "wkts_last_6",
    "wkts_last_12"
]

df = deliveries[FEATURES + ["win"]].dropna()

print("Final dataset size:", df.shape)


X = df.drop("win", axis=1)
y = df["win"]


# ======================================================
# 13. PREPROCESSING
# ======================================================
cat_cols = ["batting_team", "bowling_team", "venue", "phase"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)


# ======================================================
# 14. MODEL (OPTIMAL FOR TABULAR DATA)
# ======================================================
model = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

pipe = Pipeline(
    [
        ("prep", preprocessor),
        ("model", model)
    ]
)


# ======================================================
# 15. TRAIN / TEST
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipe.fit(X_train, y_train)

probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("Accuracy   :", round(accuracy_score(y_test, preds), 4))
print("BrierScore :", round(brier_score_loss(y_test, probs), 4))


# ======================================================
# 16. SAVE MODEL
# ======================================================
with open("model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ model.pkl saved successfully")
