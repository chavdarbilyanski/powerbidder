import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import RL.batteryEnv
from stable_baselines3 import PPO

# --- 1. Configuration & Data Loading ---

DATA_FILE_NAME = 'combine/combined_output_with_features.csv'

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

# Your storage system's parameters
STORAGE_CAPACITY_KWH = 100.0
BUY_THRESHOLD_PERCENT = 25
SELL_THRESHOLD_PERCENT = 75
MIN_CHARGE_FOR_SELL = 0.1
MAX_CHARGE_FOR_BUY = 0.9

# ----Data Loading and Preprocessing----
# CORRECTED: Added decimal=',' to correctly parse prices like '61,84'
# Use the file with the columns already added
dataset = pd.read_csv(DATA_FILE_NAME, sep=';', decimal=',')

#dataset = pd.read_csv(DATA_FILE_NAME, header=None, sep = ';', skiprows=[0],
#                    names = [DATE_COLUMN, DAY_OF_WEEK_COLUMN, MONTH_COLUMN, HOUR_COLUMN, VOLUME_COLUMN, PRICE_COLUMN])


# Rename columns for easier access
dataset.rename(columns={
    'Date': DATE_COLUMN,
    'DayOfWeek': DAY_OF_WEEK_COLUMN,
    'Month': MONTH_COLUMN,
    'Hour': HOUR_COLUMN,
    'Volume': VOLUME_COLUMN,
    'Price (EUR)': PRICE_COLUMN
}, inplace=True)


# Convert price and volume to float, handling potential errors
dataset[PRICE_COLUMN] = pd.to_numeric(dataset[PRICE_COLUMN], errors='coerce')
dataset[VOLUME_COLUMN] = pd.to_numeric(dataset[VOLUME_COLUMN], errors='coerce')

# Convert date column to datetime objects for feature engineering
dataset[DATE_COLUMN] = pd.to_datetime(dataset[DATE_COLUMN], format='%m/%d/%y', errors='coerce')

# Drop rows where conversion failed
dataset.dropna(subset=[PRICE_COLUMN, VOLUME_COLUMN, DATE_COLUMN], inplace=True)


# Add 24h rolling average price feature
dataset[PRICE_AVG24_COLUMN] = dataset[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()

# --- 2. Realistic Battery Simulation (for Target Generation) ---
# This part replaces your random generation for creating the "ground truth"
# NOTE: This is STILL a simplification for creating a target for the SUPERVISED model.
# The real dynamic simulation comes in the backtesting/RL part later.
dataset[STORAGE_CURRENT_PERCENT] = np.random.uniform(0, 1.0, size=dataset.shape[0])


# --- 3. Define the Target Variable (The "Ground Truth") ---
price_buy_threshold = np.percentile(dataset[PRICE_COLUMN], BUY_THRESHOLD_PERCENT)
price_sell_threshold = np.percentile(dataset[PRICE_COLUMN], SELL_THRESHOLD_PERCENT)

print(f"\nPrice to BUY below: {price_buy_threshold:.2f}")
print(f"Price to SELL above: {price_sell_threshold:.2f}")

def define_action(row):
    if row[PRICE_COLUMN] >= price_sell_threshold and row[STORAGE_CURRENT_PERCENT] > MIN_CHARGE_FOR_SELL:
        return "SELL"
    elif row[PRICE_COLUMN] <= price_buy_threshold and row[STORAGE_CURRENT_PERCENT] < MAX_CHARGE_FOR_BUY:
        return "BUY"
    else:
        return "HOLD"

dataset['action'] = dataset.apply(define_action, axis=1)
print("\nAction Distribution:")
print(dataset['action'].value_counts())

# --- 4. Prepare Data for Machine Learning ---
# CORRECTED: Added STORAGE_CURRENT_PERCENT to the features. This is critical.
features = [
    DAY_OF_WEEK_COLUMN,
    MONTH_COLUMN,
    HOUR_COLUMN,
    STORAGE_CURRENT_PERCENT, # Model MUST know the battery state
    VOLUME_COLUMN,
    PRICE_COLUMN,
    PRICE_AVG24_COLUMN
]
X = dataset[features]
y = dataset['action']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPORTANT: Fit the scaler ONLY on the training data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# --- 5. Train the Machine Learning Model ---
print("\n--- Training the RandomForest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model's Performance ---
print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Feature Importance ---
# The model sees scaled data, but we can still use the original feature names for clarity
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importances ---")
print(feature_importances)


# --- 8. Make a Prediction on a New, Unseen Data Point ---
# CORRECTED: The prediction section is now functional and realistic.
print("\n--- Example Prediction ---")
new_data_point = {
    DAY_OF_WEEK_COLUMN: 1, # Tuesday
    MONTH_COLUMN: 7, # July
    HOUR_COLUMN: 2, # 2 AM
    STORAGE_CURRENT_PERCENT: 0.5, # 50% full
    VOLUME_COLUMN: 1000,
    PRICE_COLUMN: 25.0,  # Very cheap price
    PRICE_AVG24_COLUMN: 105.0 # Average has been high
}

# Convert to DataFrame, ensuring the order matches the training data
new_data_df = pd.DataFrame([new_data_point])[features]

# CORRECTED: Scale the new data using the SAME scaler fitted on the training data
new_data_scaled = sc.transform(new_data_df)

prediction = model.predict(new_data_scaled)
prediction_proba = model.predict_proba(new_data_scaled)

print(f"\nNew Data: {new_data_point}")
print(f"Predicted Action: >>> {prediction[0]} <<<")
print(f"Prediction Probabilities (Buy, Hold, Sell): {model.classes_} -> {prediction_proba[0]}")

# --- 9. Backtesting Simulation with Realistic Battery State ---

print("\n--- Starting Backtesting Simulation ---")

# Simulation Parameters
INITIAL_KWH = STORAGE_CAPACITY_KWH / 2  # Start with a half-full battery
CHARGE_RATE_KW = 25                     # Max charge/discharge rate per hour
EFFICIENCY = 0.90                       # 90% round-trip efficiency (loses 10%)

# Get a clean test set from your original data (before scaling)
# This uses the same rows as X_test but with the original, unscaled values
test_data_with_actions = dataset.loc[X_test.index]
test_data_with_actions = test_data_with_actions.sort_index() # Sort by time


# Simulation state variables
profit = 0.0
current_kwh = INITIAL_KWH
transactions = []
# Loop through each hour in the test data
# for index, row in test_data_with_actions.iterrows():
#     # 1. Prepare the input for the model for the current hour
#     current_storage_percent = current_kwh / STORAGE_CAPACITY_KWH
    
#     model_input_dict = {
#         DAY_OF_WEEK_COLUMN: row[DAY_OF_WEEK_COLUMN],
#         MONTH_COLUMN: row[MONTH_COLUMN],
#         HOUR_COLUMN: row[HOUR_COLUMN],
#         STORAGE_CURRENT_PERCENT: current_storage_percent,
#         VOLUME_COLUMN: row[VOLUME_COLUMN],
#         PRICE_COLUMN: row[PRICE_COLUMN],
#         PRICE_AVG24_COLUMN: row[PRICE_AVG24_COLUMN]
#     }
    
#     input_df = pd.DataFrame([model_input_dict])[features]
#     input_scaled = sc.transform(input_df) # Use the same scaler

#     # 2. Get the model's decision
#     action = model.predict(input_scaled)[0]

#     # 3. Execute the action and update battery/profit
#     price_per_kwh = row[PRICE_COLUMN] / 1000.0 # Convert from EUR/MWh to EUR/KWH
    
#     if action == "BUY" and current_kwh < STORAGE_CAPACITY_KWH:
#         # Buy as much as possible up to the charge rate or storage capacity
#         kwh_to_buy = min(CHARGE_RATE_KW, STORAGE_CAPACITY_KWH - current_kwh)
#         cost = kwh_to_buy * price_per_kwh
        
#         profit -= cost
#         current_kwh += kwh_to_buy * EFFICIENCY # Account for charging loss
#         transactions.append({'time': index, 'action': 'BUY', 'kwh': kwh_to_buy, 'cost_profit': -cost})

#     elif action == "SELL" and current_kwh > 0:
#         # Sell as much as possible up to the discharge rate or available energy
#         kwh_to_sell = min(CHARGE_RATE_KW, current_kwh)
#         revenue = kwh_to_sell * price_per_kwh
        
#         profit += revenue
#         current_kwh -= kwh_to_sell
#         transactions.append({'time': index, 'action': 'SELL', 'kwh': kwh_to_sell, 'cost_profit': revenue})
        
#     else: # HOLD
#         transactions.append({'time': index, 'action': 'HOLD', 'kwh': 0, 'cost_profit': 0})
        
#     # Ensure battery level is within bounds
#     current_kwh = max(0, min(current_kwh, STORAGE_CAPACITY_KWH))

#     # --- Simulation Results ---
#     transaction_log = pd.DataFrame(transactions)
#     # Filter actual transaction (BUY OR SELL)
#     filtered_transaction_log = transaction_log[transaction_log['action'] != "HOLD"]
#     print(f"\nSimulation Complete. Total Profit: €{profit:.2f}")
#     print("\n--- Sample of Simulated Transactions ---")
#     print(filtered_transaction_log)
#     print("\n--- Total profit-------------------------")
#     print(profit)

print("\nRL Model HERE----- ")
# Prepare the data (use your full dataset)
rl_data = dataset.sort_index()

# Create the environment
env = RL.batteryEnv.BatteryEnv(
    historical_data=rl_data, 
    storage_capacity=STORAGE_CAPACITY_KWH, 
    charge_rate=25, 
    efficiency=0.9
)

# Instantiate and train the PPO agent
# PPO is a great, robust algorithm to start with
model_rl = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_battery_tensorboard/")
model_rl.learn(total_timesteps=len(rl_data) * 500) # Train for 5 "epochs"



# Now you can use model_rl to make predictions and run a new backtest!