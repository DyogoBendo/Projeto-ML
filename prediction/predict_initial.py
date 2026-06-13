import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 1. Load your pristine, team-agnostic numerical dataset
file_path = '../processed-data/training_data_final.csv'
df = pd.read_csv(file_path)

print(f"Total dataset loaded. Shape: {df.shape}")

# =========================================================
# 1. THE CHRONOLOGICAL SPLIT (OUT-OF-TIME VALIDATION)
# =========================================================
# CSV Line 7733 translates to Pandas Index 7731
split_index = 7731

# Training Set: Everything BEFORE the split index (Historical Seasons)
train_df = df.iloc[:split_index].copy()

# Testing Set: Everything FROM the split index onward (Latest Season)
test_df = df.iloc[split_index:].copy()

# =========================================================
# 2. SEPARATE FEATURES (X) AND TARGET (y)
# =========================================================
# We drop the final result from the X data so the model can't cheat!
# Make sure 'FTR' matches the exact name of your target column (e.g., Target, FTR)

X_train = train_df.drop(columns=['FTR'])
y_train = train_df['FTR']

X_test = test_df.drop(columns=['FTR'])
y_test = test_df['FTR']

print("\n--- Split Results ---")
print(f"Training Features (X_train): {X_train.shape}")
print(f"Training Target (y_train):   {y_train.shape}")
print(f"Testing Features (X_test):   {X_test.shape}")
print(f"Testing Target (y_test):     {y_test.shape}")

print("Training Baseline Random Forest...")

# Initialize the model 
# (n_estimators=200 means we are building 200 different decision trees)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')

# Train the model on the historical data
rf_model.fit(X_train, y_train)

# =========================================================
# 2. OUT-OF-TIME PREDICTIONS
# =========================================================
# Force the model to predict the unseen testing season
y_pred = rf_model.predict(X_test)

# Evaluate the baseline accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Baseline Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# 3. FEATURE IMPORTANCE 
# =========================================================
# Let's see which of your 421 features the algorithm actually cares about!
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Create a dataframe of the top 20 most important features
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(20)

# Print a clean, formatted list to the terminal
print("\n=========================================================")
print("TOP 20 MOST IMPORTANT FEATURES (COPY AND PASTE THIS)")
print("=========================================================")
for index, row in feature_df.iterrows():
    # Multiplies by 100 to show the importance as a percentage
    print(f"{row['Feature']:<35} | {row['Importance'] * 100:.2f}%")
print("=========================================================\n")
