import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
import torch



# Import the centralized preprocessing functions
from ml.shared.preprocessing import preprocess_data_for_model, create_observation

class BatteryStorageEnv(gym.Env):
    def __init__(self, data_path, max_battery_capacity=100.0, charge_discharge_rate=50.0):
        super(BatteryStorageEnv, self).__init__()

        # --- Data Loading and Preprocessing ---
        raw_df = pd.read_csv(data_path, sep=';', decimal=',')
        self.df = preprocess_data_for_model(raw_df)
        self.n_steps = len(self.df)

        # --- Environment Parameters ---
        self.max_capacity = max_battery_capacity
        self.rate = charge_discharge_rate
        self.current_step = 0
        self.battery_charge = 0.0
        self.efficiency = 0.95

        # --- Define Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
        
        # The shape must match the number of features from create_observation (10)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _get_obs(self):
        current_row = self.df.iloc[self.current_step]
        battery_percent = self.battery_charge / self.max_capacity
        return create_observation(current_row, battery_percent)

    def reset(self):
        self.current_step = 0
        self.battery_charge = 0.0
        return self._get_obs()

    def step(self, action):
        current_price_mwh = self.df.iloc[self.current_step]['price']
        price_kwh = current_price_mwh / 1000.0
        reward = 0

        if action == 1 and self.battery_charge < self.max_capacity: # BUY
            amount = min(self.rate, self.max_capacity - self.battery_charge)
            self.battery_charge += amount
            reward = -price_kwh * amount
        elif action == 2 and self.battery_charge > 0: # SELL
            amount = min(self.rate, self.battery_charge)
            self.battery_charge -= amount
            reward = price_kwh * amount

        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, {}

if __name__ == '__main__':
    DATA_PATH = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/3years_combined_without_volume.csv' 
    MODEL_SAVE_PATH = 'src/ml/models/battery_ppo_agent_v3' # Save model and stats
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: BatteryStorageEnv(data_path=DATA_PATH)])
    
    # Normalize the observation space - CRITICAL for good performance
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Define the PPO model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4096,
        batch_size=128,
        gamma=0.99,
        learning_rate=get_linear_fn(0.0003, 0.0001, 1.0),
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        device=device
    )

    # Train the model
    print("--- Starting Model Training ---")
    model.learn(total_timesteps=1800000, progress_bar=True) # Adjust timesteps as needed
    print("--- Model Training Complete ---")

    # Save the trained model and the normalization stats
    model.save(f"{MODEL_SAVE_PATH}.zip")
    env.save(f"{MODEL_SAVE_PATH}_stats.pkl")
    print(f"Model and stats saved to {MODEL_SAVE_PATH}.zip and {MODEL_SAVE_PATH}_stats.pkl")