import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Algorithms
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
    datasets = {
        "Raw (All Features)": "../processed-data/training_data.csv",
        "Optimal (Selection)": "../processed-data/training_data_optimal.csv",
        "PCA (Extraction)": "../processed-data/training_data_pca.csv"
    }

    # Define the models representing different ML philosophies
    models = {
        # Tree-based (Implicit Feature Selection)
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            objective='multi:softprob',
            random_state=42, 
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced',
            random_state=42, 
            n_jobs=-1
        ),
        # Geometric-based (Requires Scaling, Susceptible to Dimensionality)
        "Support Vector Machine": SVC(
            kernel='rbf',
            probability=True, # Required for Log Loss
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance', # closer neighbors have more influence
            n_jobs=-1
        )
    }

    results_list = []

    for data_name, file_path in datasets.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {data_name}.")
            continue

        print(f"\nLoading {data_name}...")
        df = pd.read_csv(file_path)

        X = df.drop(columns=['FTR'])
        y = df['FTR']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Standard split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # CRITICAL ACADEMIC STEP: Data Scaling
        # SVM and KNN rely on geometric distances. 
        # We MUST scale the data so all features have a mean of 0 and variance of 1.
        # Tree-based models (XGBoost/RF) are immune to scaling, so running them on scaled data is perfectly fine.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Evaluate Models
        for model_name, model in models.items():
            print(f"  -> Training {model_name}...")
            
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_probs = model.predict_proba(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            lloss = log_loss(y_test, y_pred_probs)

            results_list.append({
                "Dataset": data_name,
                "Model": model_name,
                "Accuracy": acc,
                "Macro F1": f1,
                "Log Loss": lloss
            })

    # ---------------------------------------------------------
    # Generate Output Reports
    # ---------------------------------------------------------
    if not results_list:
        print("No datasets were processed. Exiting.")
        return

    results_df = pd.DataFrame(results_list)
    
    print("\n=================================================================")
    print(" MODEL COMPARISON RESULTS ")
    print("=================================================================")
    print(results_df.to_string(index=False))
    print("=================================================================\n")

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance Across Data Architectures', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy (Higher is better)
    sns.barplot(ax=axes[0], data=results_df, x='Dataset', y='Accuracy', hue='Model')
    axes[0].set_title('Accuracy (Higher is Better)')
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(loc='lower right')

    # Plot 2: Macro F1 Score (Higher is better)
    sns.barplot(ax=axes[1], data=results_df, x='Dataset', y='Macro F1', hue='Model')
    axes[1].set_title('Macro F1 Score (Higher is Better)')
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend().remove() # Clean up extra legends

    # Plot 3: Log Loss (Lower is better)
    sns.barplot(ax=axes[2], data=results_df, x='Dataset', y='Log Loss', hue='Model')
    axes[2].set_title('Log Loss (Lower is Better)')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].legend().remove() # Clean up extra legends

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()