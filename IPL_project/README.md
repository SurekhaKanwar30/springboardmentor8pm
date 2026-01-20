# 🏏 IPL Win Probability Predictor

A Streamlit-based web application that predicts the win probability of IPL (Indian Premier League) cricket teams in ongoing matches using machine learning.

## 📋 Features

- **Real-time Predictions**: Get instant win probability predictions based on current match conditions
- **User-Friendly Interface**: Clean and intuitive UI for easy input of match parameters
- **Match Context**: View detailed match metrics including runs left, balls left, and run rates
- **Input Validation**: Robust input validation to ensure accurate predictions
- **Error Handling**: Comprehensive error handling for a smooth user experience

## 🛠️ Technology Stack

- **Streamlit**: Interactive web framework
- **Scikit-learn**: Machine learning model
- **Pandas**: Data manipulation and analysis
- **Python 3.8+**: Programming language

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IPL_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📝 How to Use

1. **Select Teams**: Choose the batting and bowling teams from the dropdown menus
2. **Choose Venue**: Select the match venue/city
3. **Enter Match Details**:
   - Target Score: The total runs the batting team needs to achieve
   - Current Score: Runs scored by the batting team so far
   - Overs Completed: Number of overs played (e.g., 5.3 = 5 overs 3 balls)
   - Wickets Lost: Number of wickets lost by the batting team

4. **Predict**: Click the "Predict Probability" button to get win probability estimates

## 📊 Model Details

The prediction model uses the following features:
- Batting team
- Bowling team
- Match venue
- Runs remaining
- Balls remaining
- Wickets remaining
- Current run rate
- Required run rate

## 📁 Project Structure

```
IPL_project/
├── app.py                          # Main Streamlit application
├── pipe.pkl                        # Pre-trained ML model
├── deliveries.csv                  # Match deliveries data
├── matches.csv                     # Match metadata
├── requirements.txt                # Python dependencies
├── setup.sh                        # Deployment script
├── IPL Win Probability Predictor.ipynb  # Notebook for model training
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── README.md                       # This file
```

## 🐛 Troubleshooting

- **Model file not found**: Ensure `pipe.pkl` is in the project root directory
- **Import errors**: Run `pip install -r requirements.txt` to install all dependencies
- **Port already in use**: Run `streamlit run app.py --server.port 8502` to use a different port

## 📈 Deployment

The project includes a `Procfile` for easy deployment to Heroku or similar platforms.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as an IPL win prediction machine learning project.

---

**Note**: This model is trained on historical IPL data and predictions should be used for educational and entertainment purposes.
