import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    print("Loading dataset for Feature Extraction...")
    try:
        df = pd.read_csv('../processed-data/training_data.csv')    
    except FileNotFoundError:
        print("Error: Could not find '../processed-data/training_data.csv'")
        return

    # Separate features and target
    X = df.drop(columns=['FTR'])
    y = df['FTR']
    
    print(f"Initial shape: {X.shape}")

    # Standardize the features (mean=0, variance=1) prior to PCA
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA on the fully scaled dataset to determine variance ratios
    print("Fitting PCA to determine optimal components...")
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Define the target variances requested by the professor
    target_variances = [0.90, 0.95, 0.99]

    for target_variance in target_variances:
        # Handle 100% variance carefully due to floating point precision limits
        if target_variance >= 1.00:
            optimal_components = len(cumulative_variance)
        else:
            optimal_components = np.argmax(cumulative_variance >= target_variance) + 1

        print(f"\n--- Target Variance: {target_variance*100:.0f}% ---")
        print(f"Original feature count: {X.shape[1]}")
        print(f"Components needed to retain {target_variance*100:.0f}% of variance: {optimal_components}")
        print(f"Dimensionality reduced by: {X.shape[1] - optimal_components} features")

        # Transform the dataset using the optimal number of components
        pca_optimal = PCA(n_components=optimal_components)
        X_pca = pca_optimal.fit_transform(X_scaled)

        # Reconstruct the DataFrame with principal components
        pca_columns = [f'PC_{i+1}' for i in range(optimal_components)]
        df_pca_final = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        # Reattach the target variable
        df_pca_final['FTR'] = y.values

        # Save the transformed dataset with the variance percentage in the filename
        output_path = f'../processed-data/training_data_pca_{int(target_variance*100)}.csv'
        df_pca_final.to_csv(output_path, index=False)
        print(f"PCA Extracted Dataset saved to: {output_path}")

if __name__ == "__main__":
    main()