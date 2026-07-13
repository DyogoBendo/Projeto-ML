import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import time

def main():
    print("Loading dataset for Feature Selection...")
    try:
        df = pd.read_csv('../processed-data/training_data.csv')
    except FileNotFoundError:
        print("Error: Could not find '../processed-data/training_data.csv'")
        return

    # Separate features and target
    X = df.drop(columns=['FTR'])
    y = df['FTR']

    # Encode target variable ('A', 'D', 'H' -> 0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Initial feature count: {X.shape[1]}")

    # CRITICAL: SVC and KNN require scaled data to evaluate distances properly.
    # We scale the data for the selection process, but we will save the original unscaled values.
    print("Scaling data for geometric evaluation...")
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

    # Using StratifiedKFold to explicitly treat rows as independent
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define the base estimators for each algorithm
    # SVC is now using 'rbf' exactly as it will in your final model pipeline
    models = {
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),        
    }

    print("\nStarting individual Feature Selection per algorithm...")

    for model_name, model in models.items():
        print(f"\n=======================================================")
        print(f" OPTIMIZING FEATURES FOR: {model_name}")
        print(f"=======================================================")
        start_time = time.time()        
        selector = RFECV(
            estimator=model,
            step=0.05,            # Drop bottom 5% of features each iteration
            cv=cv_strategy,
            scoring='accuracy',
            min_features_to_select=15,
            n_jobs=-1
        )
        selector.fit(X_scaled, y_encoded)
        selected_features = X.columns[selector.support_].tolist()
        optimal_num_features = selector.n_features_

        elapsed_time = time.time() - start_time
        print(f"-> Finished in {elapsed_time:.2f} seconds.")
        print(f"-> Optimal number of features found: {optimal_num_features}")
        
        # Save the optimized dataset using the ORIGINAL unscaled data
        X_optimal = X[selected_features]
        df_optimal = pd.concat([X_optimal, df['FTR']], axis=1)
        
        output_path = f'../processed-data/training_data_optimal_{model_name}.csv'
        df_optimal.to_csv(output_path, index=False)
        print(f"-> Dataset saved to: {output_path}")

    k_values = [25, 50, 100]

    print("\nStarting SelectKBest Feature Selection...")

    for k in k_values:
        print(f"\n=======================================================")
        print(f" OPTIMIZING FEATURES FOR: Top {k} Features")
        print(f"=======================================================")
        start_time = time.time()

        print(f"Method: Filter Method (SelectKBest with Mutual Information)")
        print(f"[!] Instantly calculating the top {k} features using Mutual Information statistics...")
        
        # Initialize and fit SelectKBest
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_scaled, y_encoded)
        
        selected_features = X.columns[selector.get_support()].tolist()
        optimal_num_features = len(selected_features)

        elapsed_time = time.time() - start_time
        print(f"-> Finished in {elapsed_time:.2f} seconds.")
        print(f"-> Optimal number of features found: {optimal_num_features}")
        
        # Save the optimized dataset using the ORIGINAL unscaled data
        X_optimal = X[selected_features]
        df_optimal = pd.concat([X_optimal, df['FTR']], axis=1)
        
        output_path = f'../processed-data/training_data_kbest_{k}.csv'
        df_optimal.to_csv(output_path, index=False)
        print(f"-> Dataset saved to: {output_path}")
if __name__ == "__main__":
    main()