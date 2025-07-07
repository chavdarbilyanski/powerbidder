import pandas as pd
from stable_baselines3 import PPO
import os
from django.shortcuts import render
from django.conf import settings
from .forms import CSVUploadForm
from .models import TradeSimulation
import json
import numpy as np

# Battery constraints
MAX_BATTERY_CAPACITY = 100.0  # kWh
CHARGE_DISCHARGE_RATE = 25.0  # kW per time step

def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'webbidder', 'battery_ppo_agent.zip')
    return PPO.load(model_path)

def run_inference_and_calculate(df, model):
    battery_charge = 50.0 # Assume starting from half loaded
    total_profit = 0.0
    results = []

    for index, row in df.iterrows():
        # Extract features for the RL model (adjust based on your model's observation space)
        # Example: assuming the CSV columns match the observation space
        features = row.drop('price').values.astype(np.float32)  # Convert to float32 for PPO
        # PPO model expects a numpy array for prediction
        action, _ = model.predict(features, deterministic=True)  # Get action from PPO model

        # Map action to Buy, Hold, Sell (adjust based on your model's action space)
        action_map = {0: 'Buy', 1: 'Hold', 2: 'Sell'}  # Adjust mapping as per your PPO action space
        action_str = action_map.get(action, 'Hold')  # Default to Hold if action is invalid

        price = row['price']  # Electricity price per kWh
        action_details = {
            'time_step': index,
            'action': action_str,
            'price': float(price),
            'battery_charge': battery_charge
        }

        if action_str == 'Buy' and battery_charge + CHARGE_DISCHARGE_RATE <= MAX_BATTERY_CAPACITY:
            battery_charge += CHARGE_DISCHARGE_RATE
            total_profit -= price * CHARGE_DISCHARGE_RATE  # Spend money to buy
            action_details['profit_change'] = -price * CHARGE_DISCHARGE_RATE
        elif action_str == 'Sell' and battery_charge >= CHARGE_DISCHARGE_RATE:
            battery_charge -= CHARGE_DISCHARGE_RATE
            total_profit += price * CHARGE_DISCHARGE_RATE  # Earn money by selling
            action_details['profit_change'] = price * CHARGE_DISCHARGE_RATE
        else:
            action_details['profit_change'] = 0.0  # Hold or invalid action

        action_details['battery_charge'] = battery_charge
        results.append(action_details)

    return battery_charge, total_profit, results

def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            # Save the uploaded file
            simulation = TradeSimulation.objects.create(csv_file=csv_file)
            
            # Read CSV and run inference
            df = pd.read_csv(simulation.csv_file.path)
            model = load_model()
            battery_charge, total_profit, results = run_inference_and_calculate(df, model)

            # Save results to the model
            simulation.battery_charge = battery_charge
            simulation.total_profit = total_profit
            simulation.results = json.dumps(results)
            simulation.save()

            # Render results
            return render(request, 'results.html', {
                'battery_charge': battery_charge,
                'total_profit': total_profit,
                'results': results
            })
    else:
        form = CSVUploadForm()
    
    return render(request, 'upload.html', {'form': form})