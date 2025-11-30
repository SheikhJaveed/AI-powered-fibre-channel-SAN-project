import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json

print("Starting ML model training...")

# --- Phase 2: Feature Extraction + Dataset Prep ---

# Load data
try:
    data = pd.read_csv('san_traffic.csv')
except FileNotFoundError:
    print("Error: 'san_traffic.csv' not found.")
    print("Please run 'san_traffic_simulator.py' first to generate the data.")
    exit()

print(f"Loaded {len(data)} records from san_traffic.csv.")

# Create derived features
print("Engineering new features...")
data['Total_IOPS'] = data['Read_IOPS'] + data['Write_IOPS']
data['IOPS_to_Bandwidth_Ratio'] = data['Total_IOPS'] / data['Bandwidth_MBps']
data['Latency_MA_5'] = data['Latency_ms'].rolling(window=5).mean() # Moving average of latency
data['Queue_Growth_Rate'] = data['Queue_Length'].diff().fillna(0)

# Handle NaN values created by rolling window
data = data.dropna()

# Define feature set (X) and target (y)
# We use the plan's threshold-based label as our target
y = data['Congestion']

# Define the features to be used for training
# We include our new derived features
feature_columns = [
    'Read_IOPS', 
    'Write_IOPS', 
    'Queue_Length', 
    'Bandwidth_MBps',
    'Total_IOPS',
    'IOPS_to_Bandwidth_Ratio',
    'Latency_MA_5',
    'Queue_Growth_Rate'
]
X = data[feature_columns]

print(f"Features for training: {feature_columns}")

# --- Phase 3: ML Model for Congestion Prediction ---

print("Splitting data and scaling features...")
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
# Scaling is good practice, even for Random Forest, and critical for other models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training RandomForestClassifier...")
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluate the model
print("\nModel Evaluation:")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Congested (0)', 'Congested (1)']))

# Show feature importance
print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=feature_columns)
print(importances.sort_values(ascending=False))

# Save the model, scaler, and feature columns
print("\nSaving model and scaler...")
joblib.dump(model, 'congestion_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
with open('model_columns.json', 'w') as f:
    json.dump(feature_columns, f)

print("Training complete. Model saved to 'congestion_model.joblib'.")
