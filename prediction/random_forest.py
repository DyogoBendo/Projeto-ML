import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def evaluate_best_rf():
    print("\n" + "="*60)
    print(" EVALUATING: Optimized Random Forest on KBest_50")
    print("="*60)
    
    file_path = '../processed-data/training_data_kbest_50.csv'
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        print("Make sure you have run the Feature Selection script first!")
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(file_path)
    X = df.drop(columns=['FTR'])
    y = df['FTR']

    # 'A' becomes 0, 'D' becomes 1, 'H' becomes 2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 2. Initialize Model with your Tuned Parameters
    # Random Forest DOES NOT need data scaling, so we feed it the raw data directly.
    print("Training Random Forest with optimal parameters...")
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=10,
        bootstrap=True,
        random_state=42, 
        n_jobs=-1
    )

    # 3. Train and Predict
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    # 4. Calculate Global Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    lloss = log_loss(y_test, y_pred_probs)
    
    # Calculate Macro Precision and Recall for general metrics
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate Precision and Recall per class (average=None returns an array for [A, D, H])
    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    print("\n--- GLOBAL TEST SET METRICS ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Macro Precision:  {macro_precision * 100:.2f}%")
    print(f"Macro Recall:     {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:   {f1:.4f}")
    print(f"Log Loss:         {lloss:.4f}")
    
    print("\n--- PRECISION & RECALL ANALYSIS (TRUST METRICS) ---")
    print("Precision: When the model guesses a result, what is the % chance it is actually right?")
    print("Recall: Out of all the times this result actually happened, what % did the model catch?")
    # le.inverse_transform converts [0, 1, 2] back to ['A', 'D', 'H']
    for idx, class_name in enumerate(le.classes_):
        if class_name == 'A':
            label = "Away Win (A)"
        elif class_name == 'D':
            label = "Draw (D)"
        else:
            label = "Home Win (H)"
            
        print(f" -> {label}:")
        print(f"      Precision: {precisions[idx] * 100:.2f}%")
        print(f"      Recall:    {recalls[idx] * 100:.2f}%\n")

    # 5. Create the Confusion Matrix Graphic
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("Random Forest (KBest 50 Features)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Result', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Result', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save the graphic directly to your folder!
    plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n[!] Confusion matrix saved as 'rf_confusion_matrix.png' in your current folder.")
    
    plt.show()

if __name__ == "__main__":
    evaluate_best_rf()