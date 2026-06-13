import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


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

# =========================================================
# 1. ISOLATE THE TOP 50 FEATURES
# =========================================================

# Extract the names of the Top 50 features from the Random Forest
top_50_features = feature_df.head(50)['Feature'].tolist()

print(f"Slicing dataset from {X_train.shape[1]} columns down to 50...")

# Create new lean datasets containing ONLY the pure signal
X_train_50 = X_train[top_50_features].copy()
X_test_50 = X_test[top_50_features].copy()


# CRITICAL STEP: XGBoost cannot read strings like 'H', 'D', 'A'. 
# We must use a LabelEncoder to translate them to integers (0, 1, 2)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("Training XGBoost on Top 50 Features...")

# Initialize XGBoost with specific anti-overfitting constraints
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,      # Slows down learning so it doesn't jump to conclusions
    max_depth=4,             # Prevents the trees from growing too deep and memorizing noise
    subsample=0.8,           # Only uses 80% of data per tree to increase robustness
    colsample_bytree=0.8,    # Only uses 80% of features per tree to force variety
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    n_jobs=-1
)

# Train the upgraded model on the lean data
xgb_model.fit(X_train_50, y_train_encoded)

# =========================================================
# 3. EVALUATE THE LEAN MODEL
# =========================================================
# Predict the unseen testing season
y_pred_encoded = xgb_model.predict(X_test_50)

# Translate the 0, 1, 2 predictions back into 'A', 'D', 'H'
y_pred_labels = le.inverse_transform(y_pred_encoded)

# Evaluate the new accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n--- Lean Model Evaluation (XGBoost) ---")
print(f"New Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Encode targets to 0, 1, 2
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# CRITICAL FIX: Calculate weights to penalize missing the minority class (Draws)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

print("Training Weighted XGBoost on Top 50 Features...")

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,      
    max_depth=4,             
    subsample=0.8,           
    colsample_bytree=0.8,    
    objective='multi:softprob', # Changed to softprob to get raw percentages!
    num_class=3,
    random_state=42,
    n_jobs=-1
)

# Train the model WITH the sample weights
xgb_model.fit(X_train_50, y_train_encoded, sample_weight=sample_weights)

# =========================================================
# 3. EVALUATE PROBABILITIES AND PREDICTIONS
# =========================================================
# Get the raw probabilities (e.g., [0.20, 0.30, 0.50] for A, D, H)
y_pred_probs = xgb_model.predict_proba(X_test_50)

# The hard prediction is just the class with the highest probability
y_pred_encoded = y_pred_probs.argmax(axis=1)
y_pred_labels = le.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n--- Weighted XGBoost Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Let's peek at the first 5 predictions to see the actual math
print("\n--- Inside the Brain (First 5 Matches) ---")
classes = le.classes_ # Usually ['A', 'D', 'H']
for i in range(5):
    print(f"Match {i+1} True Result: {y_test.iloc[i]}")
    for j, class_name in enumerate(classes):
        print(f"  -> Model chance of {class_name}: {y_pred_probs[i][j] * 100:.1f}%")
    print("-" * 30)

# =========================================================
# 4. ADVANCED VALUE BETTING BACKTESTER (With Safety Margin)
# =========================================================
print("\nInitializing Advanced Value Betting Simulator...")

backtest_df = X_test[['Avg_Odds_Home', 'Avg_Odds_Draw', 'Avg_Odds_Away']].copy()
backtest_df['True_Result'] = y_test.values

backtest_df['Prob_A'] = y_pred_probs[:, 0] 
backtest_df['Prob_D'] = y_pred_probs[:, 1] 
backtest_df['Prob_H'] = y_pred_probs[:, 2] 

backtest_df['Bookie_Prob_A'] = 1 / backtest_df['Avg_Odds_Away']
backtest_df['Bookie_Prob_D'] = 1 / backtest_df['Avg_Odds_Draw']
backtest_df['Bookie_Prob_H'] = 1 / backtest_df['Avg_Odds_Home']

# Calculate the actual mathematical edge (Model % - Bookie %)
backtest_df['Edge_A'] = backtest_df['Prob_A'] - backtest_df['Bookie_Prob_A']
backtest_df['Edge_D'] = backtest_df['Prob_D'] - backtest_df['Bookie_Prob_D']
backtest_df['Edge_H'] = backtest_df['Prob_H'] - backtest_df['Bookie_Prob_H']

# --- BANKROLL SETTINGS ---
flat_bet = 10.00
MINIMUM_EDGE = 0.15  # The model must have at least a 5% edge to place a bet!

total_invested = 0.00
total_returned = 0.00
bets_placed = 0
bets_won = 0

for index, row in backtest_df.iterrows():
    true_res = row['True_Result']
    
    # Put the edges in a dictionary to easily find the best one
    edges = {
        'A': {'edge': row['Edge_A'], 'odds': row['Avg_Odds_Away']},
        'D': {'edge': row['Edge_D'], 'odds': row['Avg_Odds_Draw']},
        'H': {'edge': row['Edge_H'], 'odds': row['Avg_Odds_Home']}
    }
    
    # 1. Find the single best bet in this match
    best_bet_type = max(edges, key=lambda k: edges[k]['edge'])
    best_edge_value = edges[best_bet_type]['edge']
    best_odds = edges[best_bet_type]['odds']
    
    # 2. Only bet if the best edge beats our Margin of Safety
    if best_edge_value >= MINIMUM_EDGE:
        total_invested += flat_bet
        bets_placed += 1
        
        # Check if the bet won
        if true_res == best_bet_type:
            total_returned += (flat_bet * best_odds)
            bets_won += 1

# Calculate Final ROI
profit = total_returned - total_invested
roi = (profit / total_invested) * 100 if total_invested > 0 else 0
win_rate = (bets_won / bets_placed) * 100 if bets_placed > 0 else 0

print(f"\n=========================================================")
print(f" END OF SEASON FINANCIAL REPORT (MIN EDGE: {MINIMUM_EDGE*100}%) ")
print(f"=========================================================")
print(f"Total Matches Analyzed: {len(backtest_df)}")
print(f"Total Value Bets Placed: {bets_placed}")
print(f"Bets Won: {bets_won} ({win_rate:.2f}% Win Rate)")
print(f"Total Invested: ${total_invested:.2f}")
print(f"Total Returned: ${total_returned:.2f}")
print(f"Net Profit/Loss: ${profit:.2f}")
print(f"ROI: {roi:.2f}%")
print(f"=========================================================\n")