import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def optimize_xgboost():
    print("\n" + "="*60)
    print(" OPTIMIZING: XGBoost (Focused on Away Wins)")
    print("="*60)
    
    file_path = '../processed-data/training_data_optimal.csv'
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    df = pd.read_csv(file_path)
    X = df.drop(columns=['FTR'])
    y = df['FTR']

    # 'A' becomes 0, 'D' becomes 1, 'H' becomes 2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # We must split the data BEFORE tuning so we can evaluate the final tuned model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Calculate standard balanced weights based on frequency
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    # --- BOOST AWAY WINS ---
    # 'A' is encoded as 0. We manually increase the mathematical penalty 
    # for missing an Away Win by 50% on top of the standard balancing.
    class_weights_dict[0] = class_weights_dict[0] * 1
    
    # Map weights back to every individual training sample
    sample_weights = np.array([class_weights_dict[cls] for cls in y_train])

    # Define the "Knobs and Dials" to test
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    base_model = XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Optimize for 'f1_macro'. This treats all classes equally, 
    # naturally elevating the importance of minority classes like Draws and Away wins.
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50, 
        scoring='accuracy', 
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("Initiating Grid Search with Away-Win bias (This may take a few minutes)...")
    # We pass the custom weights directly into the fit method to force the trees to adapt
    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    # Extract the absolute best model from the search
    best_model = random_search.best_estimator_
    
    # Predict on the holdout test set (Data the model has never seen)
    y_pred = best_model.predict(X_test)
    y_pred_probs = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    lloss = log_loss(y_test, y_pred_probs)

    print("\n*** XGBOOST OPTIMIZATION COMPLETE ***")
    print("Best Hyperparameters:")
    for key, value in random_search.best_params_.items():
        print(f"  -> {key}: {value}")
        
    print("\n--- FINAL TEST SET METRICS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Log Loss: {lloss:.4f}")
    
    # Create the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    
    # Plot the matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("XGBoost", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Resultado real', fontsize=12, fontweight='bold')
    plt.xlabel('Resultado predito', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    optimize_xgboost()