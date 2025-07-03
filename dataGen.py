import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration & Synthetic Data Generation ---

# Your storage system's parameters
STORAGE_CAPACITY_KWH = 100.0  # Let's assume you have a 100 KWH battery
BUY_THRESHOLD_PERCENT = 25    # Buy if price is in the bottom 25%
SELL_THRESHOLD_PERCENT = 75   # Sell if price is in the top 75%
MIN_CHARGE_FOR_SELL = 0.1     # Don't sell if battery is less than 10% full
MAX_CHARGE_FOR_BUY = 0.9      # Don't buy if battery is more than 90% full

# Generate synthetic data for 1 year (8760 hours)
num_hours = 8760
rng = pd.date_range('2023-01-01', periods=num_hours, freq='h')
df = pd.DataFrame(index=rng)

# a) Create a realistic price signal with daily and weekly cycles + noise
price_base = 100  # Base price
daily_cycle = 20 * np.sin(2 * np.pi * df.index.hour / 24)
weekly_cycle = 10 * np.sin(2 * np.pi * df.index.dayofweek / 7)
random_noise = 5 * np.random.randn(num_hours)
df['price'] = price_base + daily_cycle + weekly_cycle + random_noise
# Ensure price doesn't go below a certain floor (e.g., 10)
df['price'] = df['price'].clip(lower=10)

# b) Create the state of charge (current storage level) - for simplicity, make it random
df['current_storage_kwh'] = np.random.uniform(0, STORAGE_CAPACITY_KWH, size=num_hours)
df['current_storage_percent'] = df['current_storage_kwh'] / STORAGE_CAPACITY_KWH

print("--- Sample of Generated Data ---")
print(df.head())

# --- 2. Feature Engineering ---
# The model needs numerical features, so we create them from the index

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Create rolling average to give context to the current price
df['price_rolling_avg_24h'] = df['price'].rolling(window=24, min_periods=1).mean()

# --- 3. Define the Target Variable (The "Ground Truth") ---
# This is the logic the ML model will try to learn.

# Calculate price thresholds based on historical data
price_buy_threshold = np.percentile(df['price'], BUY_THRESHOLD_PERCENT)
price_sell_threshold = np.percentile(df['price'], SELL_THRESHOLD_PERCENT)

print(f"\nPrice to BUY below: {price_buy_threshold:.2f}")
print(f"Price to SELL above: {price_sell_threshold:.2f}")

def define_action(row):
    # SELL Condition: Price is high AND we have enough energy to sell
    if row['price'] > price_sell_threshold and row['current_storage_percent'] > MIN_CHARGE_FOR_SELL:
        return "SELL"
    # BUY Condition: Price is low AND we have space to store energy
    elif row['price'] < price_buy_threshold and row['current_storage_percent'] < MAX_CHARGE_FOR_BUY:
        return "BUY"
    # HOLD Condition: All other cases
    else:
        return "HOLD"

df['action'] = df.apply(define_action, axis=1)

# Drop rows with NaN values created by rolling average
df = df.dropna()

print("\n--- Data with Features and Target Action ---")
print(df.head())
print("\nAction Distribution:")
print(df['action'].value_counts())


# --- 4. Prepare Data for Machine Learning ---

# Define our features (X) and target (y)
features = [
    'price',
    'current_storage_percent',
    'hour',
    'day_of_week',
    'month',
    'price_rolling_avg_24h'
]
X = df[features]
y = df['action']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify ensures balanced classes in split
)


# --- 5. Train the Machine Learning Model ---

print("\n--- Training the RandomForest Classifier ---")
# We use a RandomForest, which is a powerful and robust model for this task
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")


# --- 6. Evaluate the Model's Performance ---

print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test)

# Print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["BUY", "SELL", "HOLD"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BUY", "SELL", "HOLD"], yticklabels=["BUY", "SELL", "HOLD"])
plt.xlabel('Predicted Action')
plt.ylabel('True Action')
plt.title('Confusion Matrix')
plt.show()


# --- 7. Feature Importance ---
# Let's see what the model thought was most important

feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("\n--- Feature Importances ---")
print(feature_importances)


# --- 8. Make a Prediction on a New, Unseen Data Point ---

print("\n--- Example Prediction ---")
# Let's simulate a new situation:
# Price is very low, battery is half-full, it's late at night in January
new_data = {
    'price': 25.0,  # Very cheap
    'current_storage_percent': 0.5, # 50% full
    'hour': 2, # 2 AM
    'day_of_week': 1, # Tuesday
    'month': 1, # January
    'price_rolling_avg_24h': 105.0 # Average price has been high
}

# Convert to DataFrame to match the training format
new_data_df = pd.DataFrame([new_data])

# Make the prediction
prediction = model.predict(new_data_df)
prediction_proba = model.predict_proba(new_data_df)

print(f"New Data: {new_data}")
print(f"Predicted Action: >>> {prediction[0]} <<<")
print(f"Prediction Probabilities (Buy, Hold, Sell): {prediction_proba[0]}")