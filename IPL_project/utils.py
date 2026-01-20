"""Utility functions for the IPL Win Predictor application."""

import pickle
from pathlib import Path
from typing import Tuple
import pandas as pd
import streamlit as st


def load_model(model_path: str):
    """
    Load the pre-trained model from pickle file.
    
    Args:
        model_path: Path to the pickle file containing the model
        
    Returns:
        The loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model fails to load
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def validate_inputs(
    batting_team: str,
    bowling_team: str,
    overs: float,
    score: int,
    target: int,
    wickets: int
) -> Tuple[bool, str]:
    """
    Validate the input parameters for match prediction.
    
    Args:
        batting_team: Name of the batting team
        bowling_team: Name of the bowling team
        overs: Number of overs completed
        score: Current score of batting team
        target: Target score to achieve
        wickets: Number of wickets lost
        
    Returns:
        Tuple of (is_valid, error_message)
    """
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
    if score > target + 100:
        return False, "Score seems unreasonably high compared to target."
    
    return True, ""


def calculate_metrics(
    target: int,
    score: int,
    overs: float,
    wickets: int
) -> dict:
    """
    Calculate derived cricket metrics.
    
    Args:
        target: Target score
        score: Current score
        overs: Overs completed
        wickets: Wickets lost
        
    Returns:
        Dictionary containing calculated metrics
    """
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    
    # Handle division by zero
    current_run_rate = score / overs if overs > 0 else 0
    required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    return {
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_remaining': wickets_remaining,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate
    }


def prepare_prediction_input(
    batting_team: str,
    bowling_team: str,
    city: str,
    target: int,
    score: int,
    overs: float,
    wickets: int
) -> pd.DataFrame:
    """
    Prepare the input dataframe for model prediction.
    
    Args:
        batting_team: Name of the batting team
        bowling_team: Name of the bowling team
        city: Venue city
        target: Target score
        score: Current score
        overs: Overs completed
        wickets: Wickets lost
        
    Returns:
        DataFrame ready for model prediction
    """
    metrics = calculate_metrics(target, score, overs, wickets)
    
    return pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [metrics['runs_left']],
        'balls_left': [metrics['balls_left']],
        'wickets': [metrics['wickets_remaining']],
        'total_runs_x': [target],
        'cur_run_rate': [metrics['current_run_rate']],
        'req_run_rate': [metrics['required_run_rate']]
    })


def format_probability(probability: float) -> str:
    """
    Format probability as percentage string.
    
    Args:
        probability: Probability value (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{round(probability * 100)}%"
