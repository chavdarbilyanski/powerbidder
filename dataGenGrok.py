import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

# Simulate input features
data = {
    'storage_level_kwh': np.random.uniform(0, 1000, n_samples),  # Storage capacity up to 1000 kWh
    'market_price': np.random.uniform(0.05, 0.3, n_samples),  # Price in €/kWh
    'price_forecast_next_hour': np.random.uniform(0.05, 0.3, n_samples),
    'hour': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'renewable_forecast_kwh': np.random.uniform(0, 500, n_samples),
    'demand_forecast_kwh': np.random.uniform(100, 600, n_samples),
    'state_of_charge': np.random.uniform(0, 100, n_samples),  # SoC in %
    'price_volatility': np.random.uniform(0.01, 0.1, n_samples)
}

# Simulate decision labels based on simple rules (for PoC purposes)
def assign_decision(row):
    if row['market_price'] < 0.1 and row['state_of_charge'] < 80:  # Buy if price is low and storage isn't full
        return 'buy'
    elif row['market_price'] > 0.2 and row['state_of_charge'] > 20:  # Sell if price is high and storage isn't empty
        return 'sell'
    else:
        return 'hold'

df = pd.DataFrame(data)
df['decision'] = df.apply(assign_decision, axis=1)

# 2. Prepare data for ML
X = df.drop('decision', axis=1)
y = df['decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 5. Function to predict trading decision
def predict_trading_decision(storage_level, market_price, price_forecast, hour, day_of_week,
                            renewable_forecast, demand_forecast, state_of_charge, price_volatility):
    input_data = np.array([[storage_level, market_price, price_forecast, hour, day_of_week,
                           renewable_forecast, demand_forecast, state_of_charge, price_volatility]])
    prediction = model.predict(input_data)
    return prediction[0]

# Example prediction
example_input = {
    'storage_level_kwh': 500,  # Half of 1000 kWh capacity
    'market_price': 0.15,  # Moderate price
    'price_forecast_next_hour': 0.18,  # Slightly higher price expected
    'hour': datetime.datetime.now().hour,
    'day_of_week': datetime.datetime.now().weekday(),
    'renewable_forecast_kwh': 200,  # Expected renewable generation
    'demand_forecast_kwh': 300,  # Expected demand
    'state_of_charge': 50,  # 50% SoC
    'price_volatility': 0.05  # Moderate volatility
}

decision = predict_trading_decision(**example_input)
print(f"Trading Decision: {decision}")

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)