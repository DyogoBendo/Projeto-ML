import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Import all models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
    print("\n" + "="*70)
    print(" MASTER HYPERPARAMETER TUNING SUITE (MAX ACCURACY)")
    print("="*70)

    # 1. Define the models and their specific parameter grids
    models_config = {
        "XGBoost": {
            "estimator": XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1),
            "needs_scaling": False,
            "params": {
                'model__n_estimators': [100, 200, 300, 500],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
                'model__max_depth': [3, 4, 5, 6],
                'model__subsample': [0.6, 0.8, 1.0],
                'model__colsample_bytree': [0.6, 0.8, 1.0],
                'model__gamma': [0, 0.1, 0.5, 1.0],
                'model__min_child_weight': [1, 3, 5]
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "needs_scaling": False,
            "params": {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None],
                'model__bootstrap': [True, False]
            }
        },
        "SVC": {
            "estimator": SVC(probability=True, random_state=42),
            "needs_scaling": True,
            "params": {
                'model__C': [0.1, 1, 10, 50],
                'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'model__kernel': ['rbf', 'poly'] # 'linear' is too slow for 50 iters
            }
        },
        "KNN": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "needs_scaling": True,
            "params": {
                'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
    }

    # 2. Define the datasets to test against
    base_path = '../processed-data/'
    datasets_to_check = [
        ("Raw", "training_data.csv"),
        ("KBest_25", "training_data_kbest_25.csv"),
        ("KBest_50", "training_data_kbest_50.csv"),
        ("KBest_100", "training_data_kbest_100.csv"),
        ("PCA_90", "training_data_pca_90.csv"),
        ("PCA_95", "training_data_pca_95.csv"),
        ("PCA_99", "training_data_pca_99.csv")
    ]

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_log = []

    for model_name, config in models_config.items():
        print(f"\n" + "="*50)
        print(f" COMMENCING TUNING FOR: {model_name}")
        print(f"="*50)

        # Add the model's specific optimal dataset to its checklist
        model_datasets = datasets_to_check.copy()
        model_datasets.append((f"Optimal_{model_name}", f"training_data_optimal_{model_name}.csv"))

        for data_label, filename in model_datasets:
            file_path = os.path.join(base_path, filename)
            
            if not os.path.exists(file_path):
                print(f"  [Skip] {data_label} not found ({filename})")
                continue

            print(f"\n  -> Evaluating on Dataset: {data_label}")
            
            df = pd.read_csv(file_path)
            X = df.drop(columns=['FTR'])
            y = df['FTR']

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split BEFORE tuning to evaluate real-world accuracy
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Build the pipeline depending on geometric vs tree model
            if config["needs_scaling"]:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config["estimator"])
                ])
            else:
                pipeline = Pipeline([
                    ('model', config["estimator"])
                ])

            # RandomizedSearchCV targeting Maximum Accuracy
            # Note: n_iter=20 is chosen because doing 4 models * 8 datasets = 32 tuning phases. 
            # 20 iterations keeps total runtime manageable.
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=config["params"],
                n_iter=20, 
                scoring='accuracy', 
                cv=cv_strategy,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )

            print(f"     * Searching parameters...")
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_probs = best_model.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            lloss = log_loss(y_test, y_pred_probs)

            # Clean up the best params dictionary (remove 'model__' prefix)
            clean_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}

            print(f"     * Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Log Loss: {lloss:.4f}")
            
            # Save to log
            results_log.append({
                "Model": model_name,
                "Dataset": data_label,
                "Accuracy": acc,
                "Macro_F1": f1,
                "Log_Loss": lloss,
                "Best_Params": str(clean_params)
            })

    # Save all results to a CSV so you can easily review the best combinations
    if results_log:
        results_df = pd.DataFrame(results_log)
        # Sort by best accuracy overall
        results_df = results_df.sort_values(by="Accuracy", ascending=False)
        
        output_file = '../processed-data/master_tuning_results.csv'
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print(" ALL TUNING COMPLETE! ")
        print(f" Results mapped and saved to: {output_file}")
        print(" Top 5 Combinations found:")
        print(results_df[['Model', 'Dataset', 'Accuracy']].head(5).to_string(index=False))
        print("="*70)

if __name__ == "__main__":
    main()