import os
import logging
from io import StringIO
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from django.conf import settings

# Import the centralized preprocessing functions
from ml.shared.preprocessing import preprocess_data_for_model, create_observation

logger = logging.getLogger(__name__)

# Helper class needed for loading VecNormalize stats
class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    def step(self, action): return self.observation_space.sample(), 0, False, {}
    def reset(self): return self.observation_space.sample()

def load_rl_model_and_env():
    MODEL_NAME = "battery_ppo_agent_v3" # <-- Use your new model name
    model_path = os.path.join(settings.BASE_DIR, 'powerbidder', f"{MODEL_NAME}.zip")
    stats_path = os.path.join(settings.BASE_DIR, 'powerbidder', f"{MODEL_NAME}_stats.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        raise FileNotFoundError("RL Model or stats file not found.")
    
    vec_env = DummyVecEnv([lambda: BatteryEnv()])
    vec_env = VecNormalize.load(stats_path, vec_env)
    model = PPO.load(model_path, env=vec_env)
    return model, vec_env

def run_rl_model_simulation(csv_file, max_battery_capacity, charge_discharge_rate):
    model, vec_env = load_rl_model_and_env()
    
    battery_charge, total_profit, all_results = 0.0, 0.0, []
    
    csv_data = csv_file.read().decode('utf-8')
    raw_df = pd.read_csv(StringIO(csv_data), sep=';', decimal='.')
    
    # Use the centralized preprocessing function
    processed_df = preprocess_data_for_model(raw_df)
    
    for _, row in processed_df.iterrows():
        battery_percent = (battery_charge / max_battery_capacity) if max_battery_capacity > 0 else 0.0
        
        # Use the centralized observation creation function
        obs_raw = create_observation(row, battery_percent)
        
        normalized_obs = vec_env.normalize_obs(obs_raw)
        action, _ = model.predict(normalized_obs, deterministic=True)
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(int(action), 'HOLD')
        
        battery_charge_before = battery_charge
        battery_percent_before = battery_percent
        
        price_kwh = row['price'] / 1000.0
        profit_change = 0.0
        
        if action_str == 'BUY' and battery_charge < max_battery_capacity:
            amount = min(charge_discharge_rate, max_battery_capacity - battery_charge)
            profit_change = -price_kwh * amount
            battery_charge += amount
        elif action_str == 'SELL' and battery_charge > 0:
            amount = min(charge_discharge_rate, battery_charge)
            profit_change = price_kwh * amount
            battery_charge -= amount
        
        total_profit += profit_change
        
        all_results.append({
            'timestamp': row['timestamp'].strftime('%d/%m/%Y'),
            'hour': int(row['hour']),
            'price': price_kwh,
            'action': action_str,
            'profit_change': profit_change,
            'total_profit': total_profit,
            'battery_charge': battery_charge_before,
            'battery_percent': battery_percent_before * 100.0,
        })
        
    return total_profit, all_results

def calculate_globally_optimal_profit(csv_file, max_battery_capacity, charge_discharge_rate):
    """Calculates the true maximum possible profit."""
    csv_file.seek(0)
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal='.')
    
    df.rename(columns={'Price (EUR)': 'price'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return 0.0

    buy_prices, sell_prices = [], []
    slots_per_hour = int(charge_discharge_rate)
    
    for price in df['price']:
        buy_prices.extend([price] * slots_per_hour)
        sell_prices.extend([price] * slots_per_hour)
        
    buy_prices.sort()
    sell_prices.sort(reverse=True)
    
    total_profit_mwh = 0.0
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        if sell_price > buy_price:
            total_profit_mwh += (sell_price - buy_price)
        else:
            break
            
    return total_profit_mwh / 1000.0