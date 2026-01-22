🏏 IPL Win Probability Predictor

A real-time, ball-by-ball IPL win probability prediction system built using official IPL match and delivery datasets, advanced feature engineering, and multiple machine learning models, deployed with an interactive Streamlit dashboard.

📌 Project Overview

Cricket matches are highly dynamic — win probability changes with runs, wickets, overs, pressure, and momentum.
This project predicts the probability of the batting team winning an IPL match at any point during the second innings.

🔑 Key Highlights

Uses ball-by-ball IPL data

Advanced pressure & momentum-based features

Multiple ML models for comparison

Live interactive UI using Streamlit

Real-time what-if simulation

📊 Dataset Description
1️⃣ matches.csv

Contains match-level information:

Teams

Venue

Match result

Toss decision

Winner

2️⃣ deliveries.csv

Contains ball-by-ball data:

Over & ball number

Runs scored

Wickets

Batting & bowling teams

Only official IPL datasets are used.
No external data or APIs.

⚙️ Feature Engineering (Core Strength)

The model goes beyond basic runs & wickets by engineering context-aware features:

🧠 Match State Features

Current score

Balls remaining

Wickets remaining

Runs remaining

🔥 Pressure Features

Current Run Rate (CRR)

Required Run Rate (RRR)

Pressure = RRR − CRR

📈 Momentum Features

Runs scored in last 6 & 12 balls

Wickets fallen in last 6 & 12 balls

⏱ Match Phase Awareness

Powerplay

Middle overs

Death overs

These features significantly improve prediction accuracy.
| Model               | File                 | Description                               |
| ------------------- | -------------------- | ----------------------------------------- |
| XGBoost             | `model.pkl`          | Advanced non-linear model (best accuracy) |
| Logistic Regression | `logistic_model.pkl` | Probabilistic baseline                    |
| Linear Regression   | `linear_model.pkl`   | Simple baseline                           |
| Random Forest       | `rf_model.pkl`       | Non-linear ensemble baseline              |
📈 Model Evaluation

Models are evaluated using:

Accuracy

Log Loss

Brier Score (probability quality)

XGBoost performs best due to:

Non-linear learning

Robust handling of pressure & momentum features

🖥 Streamlit Application

The Streamlit dashboard allows users to:

Select batting & bowling teams

Set target score

Adjust overs, score, wickets

Input recent momentum

Choose prediction model

View live win probability

Run what-if simulations

🔁 What-If Simulation

Users can simulate:

Extra runs in next over

Loss of wicket

Advancing overs

Win probability updates instantly.

📂 Project Structure
ipl-win-probability/
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── model.py              # XGBoost model
├── model1.py             # Linear Regression
├── model2.py             # Logistic Regression
├── model3.py             # Random Forest
│
├── utils.py              # Feature & helper functions
├── app.py                # Streamlit app
│
├── model.pkl
├── linear_model.pkl
├── logistic_model.pkl
├── rf_model.pkl
│
├── requirements.txt
└── README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train Models
python model.py
python model1.py
python model2.py
python model3.py

3️⃣ Run the App
streamlit run app.py

🧠 Technical Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

🎯 Use Cases

Live match analytics

Sports data science learning

Cricket strategy analysis

Machine learning model comparison

Portfolio / academic project

🏁 Conclusion

This project demonstrates how data science + domain knowledge can be combined to build a real-time sports analytics system.
By modeling pressure, momentum, and match context, the system delivers realistic and explainable win probability predictions.