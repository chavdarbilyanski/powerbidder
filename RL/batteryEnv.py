import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Column names 
DATE_COLUMN = 'Date'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
HOUR_COLUMN = 'Hour'
STORAGE_CURRENT_KWH = 'current_storage_kwh'
STORAGE_CURRENT_PERCENT = 'current_storage_percent'
VOLUME_COLUMN = 'Volume'
PRICE_COLUMN = 'Price (EUR)'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'

class BatteryEnv(gym.Env):
    def __init__(self, historical_data, storage_capacity, charge_rate, efficiency):
        super(BatteryEnv, self).__init__()
        
        self.data = historical_data
        self.storage_capacity = storage_capacity
        self.charge_rate = charge_rate
        self.efficiency = efficiency
        
        # ACTION SPACE: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3) 
        
        # OBSERVATION SPACE: [current_price, hour, day_of_week, battery_level_percent]
        # Define the min/max values for each part of your state
        low_bounds = np.array([0, 0, 0, 0]) 
        high_bounds = np.array([np.inf, 23, 6, 1.0])
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2 # Start half full

    def reset(self, seed=None):
        # Reset the environment to an initial state
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2
        return self._get_observation(), {}

    def _get_observation(self):
        # Get the state for the current time step
        row = self.data.iloc[self.current_step]
        battery_percent = self.current_kwh / self.storage_capacity
            
        # Example state - you will use your features here
        state = np.array([
            row[PRICE_COLUMN], 
            row[HOUR_COLUMN],
            row[DAY_OF_WEEK_COLUMN],
            battery_percent
        ], dtype=np.float32)
        return state

    def step(self, action):
        # This is where the magic happens!
        # The agent provides an 'action', and this function returns the result.
        
        current_row = self.data.iloc[self.current_step]
        price_per_kwh = current_row[PRICE_COLUMN] / 1000.0
        
        reward = 0
        
        # Action 1: BUY
        if action == 1 and self.current_kwh < self.storage_capacity:
            kwh_to_buy = min(self.charge_rate, self.storage_capacity - self.current_kwh)
            self.current_kwh += kwh_to_buy * self.efficiency
            reward = - (kwh_to_buy * price_per_kwh) # Negative reward for cost
        
        # Action 2: SELL
        elif action == 2 and self.current_kwh > 0.02:
            kwh_to_sell = min(self.charge_rate, self.current_kwh)
            self.current_kwh -= kwh_to_sell
            reward = kwh_to_sell * price_per_kwh # Positive reward for revenue
        
        # Action 0: HOLD (or illegal move)
        else:
            reward = 0 # No profit or loss for holding
            
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 # End of episode
        
        obs = self._get_observation()
        
        return obs, reward, done, False, {}
    
