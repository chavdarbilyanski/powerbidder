import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration & Synthetic Data Generation ---

DATA_FILE_NAME = 'combine/combined_output_with_features.csv'

#Column names
DATE_COLUMN = 'Date'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
HOUR_COLUMN = 'Hour'
STORAGE_CURRENT_KWH = 'current_storage_kwh',
STORAGE_CURRENT_PERCENT = 'current_storage_percent',
VOLUME_COLUMN = 'Volume'
PRICE_COLUMN = 'Price (EUR)'

PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'

# Your storage system's parameters
STORAGE_CAPACITY_KWH = 100.0  # Let's assume you have a 100 KWH battery
BUY_THRESHOLD_PERCENT = 25    # Buy if price is in the bottom 25%
SELL_THRESHOLD_PERCENT = 75   # Sell if price is in the top 75%
MIN_CHARGE_FOR_SELL = 0.1     # Don't sell if battery is less than 10% full
MAX_CHARGE_FOR_BUY = 0.9      # Don't buy if battery is more than 90% full

#----My code
dataset = pd.read_csv(DATA_FILE_NAME, header=None, sep = ';', skiprows=[0],
                    names = [DATE_COLUMN, DAY_OF_WEEK_COLUMN, MONTH_COLUMN, HOUR_COLUMN, VOLUME_COLUMN, PRICE_COLUMN])

# Convert to float using astype()
dataset[PRICE_COLUMN] = dataset[PRICE_COLUMN].astype(float)

# Convert to Unix timestamp seconds
dataset[DATE_COLUMN] = pd.to_datetime(dataset[DATE_COLUMN])
dataset[DATE_COLUMN] = (dataset[DATE_COLUMN] - pd.Timestamp("1970-01-01"))
dataset[DATE_COLUMN] = dataset[DATE_COLUMN].astype(int)

#Add 24h rolling avg
dataset[PRICE_AVG24_COLUMN] = dataset[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()


# b) Create the state of charge (current storage level) - for simplicity, make it random

dataset.insert(3, STORAGE_CURRENT_KWH, np.random.uniform(0, STORAGE_CAPACITY_KWH, size=dataset.shape[0]))
dataset.insert(4, STORAGE_CURRENT_PERCENT, dataset[STORAGE_CURRENT_KWH] / STORAGE_CAPACITY_KWH)

# --- 3. Define the Target Variable (The "Ground Truth") ---
# This is the logic the ML model will try to learn.

# Calculate price thresholds based on historical data
price_buy_threshold = np.percentile(dataset[PRICE_COLUMN], BUY_THRESHOLD_PERCENT)
price_sell_threshold = np.percentile(dataset[PRICE_COLUMN], SELL_THRESHOLD_PERCENT)

print(f"\nPrice to BUY below: {price_buy_threshold:.2f}")
print(f"Price to SELL above: {price_sell_threshold:.2f}")

def define_action(row):
    # SELL Condition: Price is high AND we have enough energy to sell
    if row[PRICE_COLUMN] > price_sell_threshold and row[STORAGE_CURRENT_PERCENT] > MIN_CHARGE_FOR_SELL:
        return "SELL"
    # BUY Condition: Price is low AND we have space to store energy
    elif row[PRICE_COLUMN] < price_buy_threshold and row[STORAGE_CURRENT_PERCENT] < MAX_CHARGE_FOR_BUY:
        return "BUY"
    # HOLD Condition: All other cases
    else:
        return "HOLD"

dataset['action'] = dataset.apply(define_action, axis=1)

# Drop rows with NaN values created by rolling average
# Maybe not needed
dataset = dataset.dropna()

print("\nAction Distribution:")
print(dataset['action'].value_counts())


# --- 4. Prepare Data for Machine Learning ---

# Define our features (X) and target (y)
features = [
    DATE_COLUMN,
    DAY_OF_WEEK_COLUMN,
    MONTH_COLUMN,
    HOUR_COLUMN,
    #STORAGE_CURRENT_KWH,
    #STORAGE_CURRENT_PERCENT,
    VOLUME_COLUMN,
    PRICE_COLUMN,
    PRICE_AVG24_COLUMN 
]
X = dataset[features]
y = dataset['action']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify ensures balanced classes in split
)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
    DATE_COLUMN:7/7/2025, #7th of July
    PRICE_COLUMN: 25.0,  # Very cheap
    STORAGE_CURRENT_PERCENT: 0.5, # 50% full
    VOLUME_COLUMN: 1000, # some volume
    HOUR_COLUMN: 2, # 2 AM
    DAY_OF_WEEK_COLUMN: 1, # Tuesday
    MONTH_COLUMN: 1, # January
    PRICE_AVG24_COLUMN: 105.0 # Average price has been high
}

# Convert to DataFrame to match the training format
new_data_df = pd.DataFrame([new_data])

# Make the prediction
prediction = model.predict(new_data_df)
prediction_proba = model.predict_proba(new_data_df)

print(f"New Data: {new_data}")
print(f"Predicted Action: >>> {prediction[0]} <<<")
print(f"Prediction Probabilities (Buy, Hold, Sell): {prediction_proba[0]}")