import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

    # Determine the number of components required to retain 95% of the variance
    target_variance = 0.95
    optimal_components = np.argmax(cumulative_variance >= target_variance) + 1

    print(f"Original feature count: {X.shape[1]}")
    print(f"Components needed to retain {target_variance*100}% of variance: {optimal_components}")
    print(f"Dimensionality reduced by: {X.shape[1] - optimal_components} features")

    # Transform the dataset using the optimal number of components
    pca_optimal = PCA(n_components=optimal_components)
    X_pca = pca_optimal.fit_transform(X_scaled)

    # Reconstruct the DataFrame with principal components
    pca_columns = [f'PC_{i+1}' for i in range(optimal_components)]
    df_pca_final = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    # Reattach the target variable
    df_pca_final['FTR'] = y.values

    # Save the transformed dataset
    output_path = '../processed-data/training_data_pca.csv'
    df_pca_final.to_csv(output_path, index=False)
    print(f"PCA Extracted Dataset saved to: {output_path}")

    # Generate the cumulative explained variance plot (Scree Plot)
    print("Generating PCA variance graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-')
    plt.axhline(y=target_variance, color='r', linestyle='--', label=f'Limiar de variância acumulada (95%)')
    plt.axvline(x=optimal_components, color='g', linestyle='--', label=f'Número de componentes: {optimal_components}')
    
    plt.scatter(optimal_components, target_variance, color='black', s=70, zorder=5, label='Ponto de Interseção')

    plt.title('PCA - Variância acumulada pelo número de componentes')
    plt.xlabel('Número de componentes')
    plt.ylabel('Variância acumulada (%)')

    # Force Y-axis to step by 0.1 (10%) and format as percentages
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()