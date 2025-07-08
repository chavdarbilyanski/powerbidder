import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from django.shortcuts import render
from django.conf import settings
from .forms import CSVUploadForm
import numpy as np
from datetime import datetime
import gym
from gym import spaces
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Battery constraints
MAX_BATTERY_CAPACITY = 3900.0  # kWh
CHARGE_DISCHARGE_RATE = 1000.0  # kW per time step
MODEL_FILE_NAME = "battery_ppo_agent_v2.zip"
STATS_FILE_NAME = "vec_normalize_stats_v2.pkl"
PRICE_COLUMN = "price"
DATE_COLUMN = "timestamp"
HOUR_COLUMN = "hour"
INITIAL_BATTERY_PERCENT = 0.0  # Initial battery charge as a fraction (0.0 = empty)

# Environment definition for normalization (using OpenAI Gym)
class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
    
    def step(self, action):
        return self.observation_space.sample(), 0, False, {}
    
    def reset(self):
        return self.observation_space.sample()

def load_model_and_env():
    logger.debug("Loading model and environment...")
    model_path = os.path.join(settings.BASE_DIR, 'webbidder', MODEL_FILE_NAME)
    stats_path = os.path.join(settings.BASE_DIR, 'webbidder', STATS_FILE_NAME)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(stats_path):
        logger.error(f"Stats file not found at {stats_path}")
        raise FileNotFoundError(f"Stats file not found at {stats_path}")
    
    vec_env = DummyVecEnv([lambda: BatteryEnv()])
    vec_env = VecNormalize.load(stats_path, vec_env)
    logger.debug("Normalization stats loaded.")
    
    model = PPO.load(model_path, env=vec_env)
    logger.debug("PPO model loaded.")
    return model, vec_env

def preprocess_csv(csv_file):
    logger.debug("Preprocessing CSV...")
    try:
        df = pd.read_csv(csv_file, sep=';', decimal='.')
        logger.debug(f"CSV loaded. Columns: {df.columns.tolist()}, Rows: {len(df)}")
        
        # Rename columns
        df.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Date': DATE_COLUMN, 'Hour': HOUR_COLUMN}, inplace=True)
        
        # Validate required columns
        required_columns = [DATE_COLUMN, HOUR_COLUMN, PRICE_COLUMN]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {', '.join(required_columns)}")
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
        # Convert price to numeric (keep in €/MWh for model)
        df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')
        logger.debug(f"Price column (€/MWh): {df[PRICE_COLUMN].head().tolist()}")
        
        # Convert timestamp to datetime
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='%d/%m/%Y', errors='coerce')
        logger.debug(f"Timestamp column converted: {df[DATE_COLUMN].head().tolist()}")
        
        # Convert hour to numeric
        df[HOUR_COLUMN] = pd.to_numeric(df[HOUR_COLUMN], errors='coerce')
        logger.debug(f"Hour column converted: {df[HOUR_COLUMN].head().tolist()}")
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        if df.empty:
            logger.error("CSV is empty after dropping NaN values")
            raise ValueError("CSV is empty after preprocessing")
        logger.debug(f"Rows after dropping NaN: {len(df)}")
        
        # Enrich with DayOfWeek and Month
        df['DayOfWeek'] = df[DATE_COLUMN].apply(lambda x: x.weekday())
        df['Month'] = df[DATE_COLUMN].apply(lambda x: x.month)
        logger.debug(f"Enriched columns: DayOfWeek={df['DayOfWeek'].head().tolist()}, Month={df['Month'].head().tolist()}")
        
        # Calculate 24-hour rolling average for price (in €/MWh)
        df['price_rolling_avg_24h'] = df[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()
        logger.debug(f"Price rolling avg (€/MWh): {df['price_rolling_avg_24h'].head().tolist()}")
        
        # Add synthetic index to handle duplicate timestamps
        df['temp_index'] = range(len(df))
        logger.debug(f"DataFrame shape: {df.shape}")
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocess_csv: {str(e)}")
        raise

def run_inference_and_calculate(df, model, vec_env):
    logger.debug("Running inference...")
    battery_charge = INITIAL_BATTERY_PERCENT * MAX_BATTERY_CAPACITY
    total_profit = 0.0
    results = []

    for _, row in df.iterrows():
        time_step = row['temp_index']
        timestamp = row[DATE_COLUMN]
        hour = row[HOUR_COLUMN]  # Keep 1-24 for display
        hour_model = row[HOUR_COLUMN] - 1  # Convert to 0-23 for model
        weekday = row['DayOfWeek']
        month = row['Month']
        price_mwh = row[PRICE_COLUMN]  # Price in €/MWh for model
        price_kwh = price_mwh / 1000.0  # Convert to €/kWh for profit and display
        logger.debug(f"Processing: time_step={time_step}, timestamp={timestamp}, hour={hour}, price_mwh={price_mwh}, price_kwh={price_kwh}")

        # Battery state BEFORE action
        battery_percent = battery_charge / MAX_BATTERY_CAPACITY

        # Use €/MWh for model input
        obs_raw = np.array([
            price_mwh,
            hour_model,
            weekday,
            month,
            row['price_rolling_avg_24h'],
            battery_percent
        ], dtype=np.float32)
        logger.debug(f"Raw observation (price in €/MWh): {obs_raw}")

        normalized_obs = vec_env.normalize_obs(np.array([obs_raw]))
        logger.debug(f"Normalized observation: {normalized_obs}")

        action, _ = model.predict(normalized_obs, deterministic=True)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map.get(action[0], 'HOLD')
        logger.debug(f"Action: {action_str}")

        # Create action_details with state BEFORE action
        action_details = {
            'time_step': time_step,
            'timestamp': timestamp.strftime('%d/%m/%Y'),  # Format as DD/MM/YYYY
            'hour': int(hour),  # Display hour (1-24)
            'action': action_str,
            'price': price_kwh,  # Display in €/kWh
            'battery_charge': battery_charge,
            'battery_percent': battery_percent * 100.0,  # Convert to percentage
            'price_rolling_avg_24h': row['price_rolling_avg_24h'] / 1000.0,  # Display in €/kWh
            'profit_change': 0.0,  # Initialize
            'total_profit': total_profit
        }

        # Apply action with max function for battery constraints
        if action_str == 'BUY' and battery_charge < MAX_BATTERY_CAPACITY:
            amount = min(CHARGE_DISCHARGE_RATE, MAX_BATTERY_CAPACITY - battery_charge)
            action_details['profit_change'] = -price_kwh * amount
            total_profit += action_details['profit_change']
            battery_charge += amount
            logger.debug(f"BUY: amount={amount}, battery_charge={battery_charge}, profit_change={action_details['profit_change']}")
        elif action_str == 'SELL' and battery_charge > 0:
            amount = min(CHARGE_DISCHARGE_RATE, battery_charge)
            action_details['profit_change'] = price_kwh * amount
            total_profit += action_details['profit_change']
            battery_charge -= amount
            logger.debug(f"SELL: amount={amount}, battery_charge={battery_charge}, profit_change={action_details['profit_change']}")
        else:
            action_details['profit_change'] = 0.0
            logger.debug(f"HOLD or invalid action: action={action_str}, battery_charge={battery_charge}")

        results.append(action_details)

    logger.debug(f"Final: battery_charge={battery_charge}, total_profit={total_profit}, rows_processed={len(results)}")
    return battery_charge, total_profit, results

def upload_csv(request):
    logger.debug(f"Received request: method={request.method}")
    if request.method == 'POST':
        logger.debug("Processing POST request")
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            logger.debug("Form is valid")
            csv_file = request.FILES['csv_file']
            logger.debug(f"Uploaded file: {csv_file.name}")
            try:
                df = preprocess_csv(csv_file)
                logger.debug(f"DataFrame shape: {df.shape}")
                
                model, vec_env = load_model_and_env()
                battery_charge, total_profit, results = run_inference_and_calculate(df, model, vec_env)

                logger.debug("Rendering results.html")
                return render(request, 'results.html', {
                    'battery_charge': battery_charge,
                    'battery_percent': (battery_charge / MAX_BATTERY_CAPACITY) * 100.0,
                    'total_profit': total_profit,
                    'results': results
                })
            except Exception as e:
                logger.error(f"Error in upload_csv: {str(e)}")
                return render(request, 'upload.html', {
                    'form': form,
                    'error': f"Error processing CSV: {str(e)}"
                })
        else:
            logger.debug(f"Form invalid: {form.errors}")
            return render(request, 'upload.html', {
                'form': form,
                'error': f"Form invalid: {form.errors}"
            })
    else:
        logger.debug("Rendering upload.html for GET request")
        form = CSVUploadForm()
        return render(request, 'upload.html', {'form': form})