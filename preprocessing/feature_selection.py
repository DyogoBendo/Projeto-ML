import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


print("Loading dataset for Feature Selection...")
try:
    df = pd.read_csv('../processed-data/training_data.csv')
except FileNotFoundError:
    print("Error: Could not find '../processed-data/training_data.csv'")    

# Separate features and target
X = df.drop(columns=['FTR'])
y = df['FTR']

# Encode target variable ('A', 'D', 'H' -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Initial feature count: {X.shape[1]}")

print("Initializing RFECV with StratifiedKFold (Independent rows assumption)...")

# Using StratifiedKFold with shuffle=True to explicitly treat rows as independent
# while preserving the distribution of target classes in each fold.
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the base estimator
base_model = XGBClassifier(
    n_estimators=100, 
    max_depth=3, 
    learning_rate=0.1, 
    random_state=42, 
    n_jobs=-1
)

# Initialize the Recursive Feature Elimination with Cross-Validation
selector = RFECV(
    estimator=base_model,
    step=0.05,            # Drop the bottom 5% of features iteratively
    cv=cv_strategy,
    scoring='accuracy',
    min_features_to_select=15,
    n_jobs=-1
)

print("Fitting RFECV (This may take several minutes)...")
selector.fit(X, y_encoded)

optimal_num_features = selector.n_features_
selected_features = X.columns[selector.support_].tolist()

print("\nFeature Selection Complete")
print(f"Optimal number of features found: {optimal_num_features}")

# Save the optimized dataset
X_optimal = X[selected_features]
df_optimal = pd.concat([X_optimal, df['FTR']], axis=1)

output_path = '../processed-data/training_data_optimal.csv'
df_optimal.to_csv(output_path, index=False)
print(f"Optimal Dataset saved to: {output_path}")

plt.figure(figsize=(10, 6))

# Extract mean test scores from the cross-validation
cv_results = selector.cv_results_['mean_test_score']

# Calculate the approximate number of features at each step
num_features_start = X.shape[1]
features_at_step = [
    max(15, num_features_start - int(num_features_start * 0.05) * i) 
    for i in range(len(cv_results))
]

# Plotting the evaluation curve (reversing arrays so x-axis grows left-to-right)
plt.plot(features_at_step[::-1], cv_results[::-1], marker='o', linestyle='-')
plt.axvline(x=optimal_num_features, color='red', linestyle='--', label=f'Optimal: {optimal_num_features} Features')

plt.title('Recursive Feature Elimination (Stratified CV)')
plt.xlabel('Number of Features Evaluated')
plt.ylabel('Cross-Validation Accuracy')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
