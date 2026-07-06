import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

# Algorithms
from xgboost import XGBClassifier
from sklearn.svm import SVC

def main():
    # Define exactly which model runs on which dataset
    target_configs = [
        {
            "Dataset_Name": "Optimal (Selection)",
            "File_Path": "../processed-data/training_data_optimal.csv",
            "Model_Name": "XGBoost",
            "Model": XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=4, 
                objective='multi:softprob',
                random_state=42, 
                n_jobs=-1
            )
        },
        {
            "Dataset_Name": "PCA (Extraction)",
            "File_Path": "../processed-data/training_data_pca.csv",
            "Model_Name": "Support Vector Machine",
            "Model": SVC(
                kernel='rbf',
                probability=True, 
                random_state=42
            )
        }
    ]

    results_list = []

    # Set the theme for the confusion matrices
    sns.set_theme(style="white")

    for i, config in enumerate(target_configs):
        data_name = config["Dataset_Name"]
        file_path = config["File_Path"]
        model_name = config["Model_Name"]
        model = config["Model"]

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {model_name} on {data_name}.")
            continue

        print("\n" + "="*60)
        print(f" EVALUATING: {model_name} on {data_name}")
        print("="*60)
        
        df = pd.read_csv(file_path)

        # Separate Features and Target
        X = df.drop(columns=['FTR'])
        y = df['FTR']

        # Encode Target (A=0, D=1, H=2)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Standard split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # CRITICAL: Scaling is mandatory for SVM, but doesn't hurt XGBoost
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        print(f" -> Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Predict
        print(f" -> Generating predictions...")
        y_pred = model.predict(X_test_scaled)
        y_pred_probs = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        lloss = log_loss(y_test, y_pred_probs)

        # Generate and format the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Format the confusion matrix into a readable Pandas DataFrame
        # le.classes_ will automatically map back to 'A', 'D', 'H'
        cm_df = pd.DataFrame(
            cm, 
            index=le.classes_, 
            columns=le.classes_
        )

        print("\n--- PERFORMANCE METRICS ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print(f"Log Loss: {lloss:.4f}")
        
        print("\n--- CONFUSION MATRIX ---")
        print(cm_df.to_string())
        print("-" * 60)

        # Draw the confusion matrix graphic with a color gradient in a new figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title(f"{model_name}", fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Resultado real', fontsize=12, fontweight='bold')
        plt.xlabel('Resultado predito', fontsize=12, fontweight='bold')
        plt.tight_layout()

        results_list.append({
            "Dataset": data_name,
            "Model": model_name,
            "Accuracy": acc,
            "Macro F1": f1,
            "Log Loss": lloss
        })

    if not results_list:
        print("No datasets were processed. Exiting.")
        return

    results_df = pd.DataFrame(results_list)
    
    print("\n=================================================================")
    print(" FINAL SUMMARY")
    print("=================================================================")
    print(results_df.to_string(index=False))
    print("=================================================================\n")

    # Display all generated figures at once
    plt.show()

if __name__ == "__main__":
    main()