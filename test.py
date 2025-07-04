import pandas as pd


transactions = []

transactions.append({'time': 1, 'action': 'SELL', 'kwh': 100, 'cost_profit': 1})
transactions.append({'time': 2, 'action': 'HOLD', 'kwh': 200, 'cost_profit': 2})
transactions.append({'time': 3, 'action': 'BUY', 'kwh': 400, 'cost_profit': 3})

# Convert to DataFrame
df = pd.DataFrame(transactions)
# Filter for action == "SELL"
filtered_df = df[df['action'] != 'HOLD']
print(filtered_df)