# src/ml/shared/preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame and prepares it with all necessary features for the
    RL model's observation space. This function will be used by both the
    training environment and the inference service.

    Args:
        df: A DataFrame that must contain 'Price (EUR)', 'Date', and 'Hour' columns.

    Returns:
        A new DataFrame with all engineered features.
    """
    # 1. Rename and clean columns
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%y', errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    
    # Drop rows where essential data is missing
    df.dropna(subset=['price', 'timestamp', 'hour'], inplace=True)
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning essential columns.")

    # 2. Engineer Time-based Features
    df['dayofweek'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month

    # 3. Engineer Cyclical Features (more robust for ML models)
    # Note: We use the raw hour (1-24) and dayofweek (0-6) for these calculations
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 4. Engineer Price-based Features
    df['price_rolling_avg_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
    df['price_volatility_24h'] = df['price'].rolling(window=24, min_periods=1).std().fillna(0)
    
    return df

def create_observation(row: pd.Series, battery_percent: float) -> np.ndarray:
    """
    Constructs a single observation numpy array from a row of preprocessed data.
    The order of features here defines the observation space.
    """
    observation = np.array([
        row['price'],
        row['price_rolling_avg_24h'],
        row['price_volatility_24h'],
        row['hour_sin'],
        row['hour_cos'],
        row['dayofweek_sin'],
        row['dayofweek_cos'],
        row['month_sin'],
        row['month_cos'],
        battery_percent
    ], dtype=np.float32)
    
    return observation