import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------- LOAD DATA ----------------
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

matches = matches[matches["result"] == "normal"]

deliveries["is_wicket"] = deliveries["player_dismissed"].notna().astype(int)

deliveries = deliveries.merge(
    matches[["id", "venue", "team1", "team2", "winner"]],
    left_on="match_id",
    right_on="id",
    how="inner"
)

# ---------------- FEATURE ENGINEERING ----------------
deliveries["ball_number"] = deliveries["over"] * 6 + deliveries["ball"]
deliveries["current_score"] = deliveries.groupby("match_id")["total_runs"].cumsum()
deliveries["wickets_fallen"] = deliveries.groupby("match_id")["is_wicket"].cumsum()

deliveries["balls_remaining"] = 120 - deliveries["ball_number"]
deliveries["wickets_remaining"] = 10 - deliveries["wickets_fallen"]

first_innings = (
    deliveries[deliveries["inning"] == 1]
    .groupby("match_id")["current_score"]
    .max()
)

deliveries = deliveries.merge(
    first_innings.rename("first_innings_score"),
    on="match_id",
    how="left"
)

deliveries = deliveries[deliveries["inning"] == 2]
deliveries["target"] = deliveries["first_innings_score"] + 1
deliveries["runs_remaining"] = deliveries["target"] - deliveries["current_score"]

deliveries["current_run_rate"] = deliveries["current_score"] * 6 / deliveries["ball_number"].clip(lower=1)
deliveries["required_run_rate"] = deliveries["runs_remaining"] * 6 / deliveries["balls_remaining"].clip(lower=1)
deliveries["pressure"] = deliveries["required_run_rate"] - deliveries["current_run_rate"]

for w in [6, 12]:
    deliveries[f"runs_last_{w}"] = deliveries.groupby("match_id")["total_runs"].rolling(w).sum().reset_index(0, drop=True)
    deliveries[f"wkts_last_{w}"] = deliveries.groupby("match_id")["is_wicket"].rolling(w).sum().reset_index(0, drop=True)

def phase(over):
    if over < 6: return "powerplay"
    elif over < 15: return "middle"
    return "death"

deliveries["phase"] = deliveries["over"].apply(phase)

deliveries["win"] = (deliveries["batting_team"] == deliveries["winner"]).astype(int)

FEATURES = [
    "batting_team","bowling_team","venue","phase",
    "current_score","balls_remaining","wickets_remaining",
    "runs_remaining","current_run_rate","required_run_rate",
    "pressure","runs_last_6","runs_last_12",
    "wkts_last_6","wkts_last_12"
]

df = deliveries[FEATURES + ["win"]].dropna()

X = df.drop("win", axis=1)
y = df["win"]

# ---------------- PIPELINE ----------------
cat_cols = ["batting_team","bowling_team","venue","phase"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = LinearRegression()

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
preds = np.clip(preds, 0, 1)

print("Linear Regression RMSE:", mean_squared_error(y_test, preds, squared=False))

with open("linear_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ linear_model.pkl saved")
