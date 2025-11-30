import pandas as pd
import numpy as np
import joblib
import json

print("Starting SAN traffic optimization simulation...")

# --- Load Model and Data ---
try:
    model = joblib.load('congestion_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    # Load the "before" data to simulate a live feed
    data = pd.read_csv('san_traffic.csv')

except FileNotFoundError:
    print("Error: Model files or 'san_traffic.csv' not found.")
    print("Please run 'san_traffic_simulator.py' and 'ml_model_trainer.py' first.")
    exit()

print("Loaded trained model, scaler, and 'san_traffic.csv' data.")
print(f"Model expects features: {model_columns}")

# --- Phase 4: Traffic Management Module ---

optimized_data = []
# Store last 5 latency values for moving average calculation
latency_window = list(data['Latency_ms'].iloc[0:5])
queue_window = list(data['Queue_Length'].iloc[0:5])

print("Simulating real-time optimization...")
for i in range(len(data)):
    if i < 5: 
        # Skip first few rows where we can't calculate features
        optimized_data.append(data.iloc[i].to_dict())
        continue

    # Get current (new) row of data
    current_row = data.iloc[i].to_dict()

    # --- Feature Engineering for Live Data ---
    # Create the same features the model was trained on
    features = {}
    features['Read_IOPS'] = current_row['Read_IOPS']
    features['Write_IOPS'] = current_row['Write_IOPS']
    features['Queue_Length'] = current_row['Queue_Length']
    features['Bandwidth_MBps'] = current_row['Bandwidth_MBps']
    
    features['Total_IOPS'] = features['Read_IOPS'] + features['Write_IOPS']
    features['IOPS_to_Bandwidth_Ratio'] = features['Total_IOPS'] / features['Bandwidth_MBps']
    
    # Use the window to calculate rolling/diff features
    features['Latency_MA_5'] = np.mean(latency_window)
    features['Queue_Growth_Rate'] = features['Queue_Length'] - queue_window[-1]

    # Convert to DataFrame for scaling
    features_df = pd.DataFrame([features], columns=model_columns)
    
    # Scale the features
    features_scaled = scaler.transform(features_df)

    # --- Prediction ---
    prediction = model.predict(features_scaled)[0]
    
    # --- Optimization Logic ---
    optimized_row = current_row.copy()
    optimized_row['Predicted_Congestion'] = prediction
    optimized_row['Action_Taken'] = 'None'

    if prediction == 1:
        # If congestion is predicted, trigger load balancing
        optimized_row['Action_Taken'] = 'Throttled Writes'
        
        # Simple optimization: Reduce write IOPS and cap queue
        optimized_row['Write_IOPS'] = int(optimized_row['Write_IOPS'] * 0.7) # Reduce by 30%
        optimized_row['Queue_Length'] = min(optimized_row['Queue_Length'], 30) # Cap queue
        
        # Recalculate latency based on optimization (simplified model)
        # In a real sim, this would be complex. Here, we just reduce it.
        optimized_row['Latency_ms'] = max(2, optimized_row['Latency_ms'] * 0.3)
        optimized_row['Congestion'] = 1 if optimized_row['Latency_ms'] > 10 else 0

    optimized_data.append(optimized_row)

    # Update windows for next iteration
    # Use the *optimized* latency for the next step's calculation
    latency_window.pop(0)
    latency_window.append(optimized_row['Latency_ms'])
    queue_window.pop(0)
    queue_window.append(optimized_row['Queue_Length'])

# --- Save Results ---
optimized_df = pd.DataFrame(optimized_data)
optimized_df.to_csv('optimized_traffic.csv', index=False)

print("\nOptimization simulation complete.")
print(f"Original congestion events: {data['Congestion'].sum()}")
print(f"Optimized congestion events: {optimized_df['Congestion'].sum()}")
print("Optimized data saved to 'optimized_traffic.csv'.")
