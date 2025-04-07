import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def run_descriptive_models(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check dataframe size and print information
    print(f"DataFrame shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Optimize memory usage for numerical columns
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    print(f"Optimized memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Select numerical features
    features = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).drop(columns=['SalePrice'], errors='ignore')
    
    # Sample features if too many columns
    max_features = 100  # Adjust based on available memory
    if features.shape[1] > max_features:
        print(f"Too many features ({features.shape[1]}). Selecting {max_features} features with highest variance.")
        # Calculate variance for non-NaN values
        variances = features.var(skipna=True).sort_values(ascending=False)
        selected_columns = variances.index[:max_features]
        features = features[selected_columns]
    
    print(f"Feature matrix shape for analysis: {features.shape}")
    
    # Check for NaN values
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values. Imputing missing values with median.")
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(features)
    else:
        features_imputed = features.values
    
    # Scale the data
    print("Scaling features...")
    features_scaled = StandardScaler().fit_transform(features_imputed.astype(np.float32))
    
    # PCA with timing
    print("Running PCA...")
    import time
    start_time = time.time()
    
    # Use lower n_components for faster computation
    n_components = min(2, features_scaled.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)
    
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    
    # Create a smaller dataframe with just what we need
    df_pca = pd.DataFrame({
        'PCA1': principal_components[:, 0],
        'PCA2': principal_components[:, 1] if n_components > 1 else np.zeros(len(principal_components))
    })

    # Generate PCA plot
    print("Generating PCA visualization...")
    plt.figure(figsize=(8, 6))
    # Sample the data if it's too large for plotting
    sample_size = min(10000, len(df_pca))
    df_pca_sample = df_pca.sample(sample_size, random_state=42) if len(df_pca) > sample_size else df_pca
    sns.scatterplot(x='PCA1', y='PCA2', data=df_pca_sample)
    plt.title('PCA - Projection on First Two Components')
    plt.savefig(os.path.join(output_dir, 'pca_scatter.png'))
    plt.close()

    # K-Means Clustering
    print("Running K-Means clustering...")
    start_time = time.time()
    
    # Use a smaller subset for clustering if dataset is very large
    max_kmeans_samples = 50000
    if features_scaled.shape[0] > max_kmeans_samples:
        print(f"Dataset too large for K-Means. Sampling {max_kmeans_samples} points.")
        indices = np.random.choice(features_scaled.shape[0], max_kmeans_samples, replace=False)
        features_for_kmeans = features_scaled[indices]
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters_sample = kmeans.fit_predict(features_for_kmeans)
        
        # Predict clusters for visualization sample
        sample_indices = df_pca_sample.index
        features_for_pred = features_scaled[sample_indices]
        df_pca_sample['Cluster'] = kmeans.predict(features_for_pred)
    else:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_pca['Cluster'] = kmeans.fit_predict(features_scaled)
        df_pca_sample = df_pca.sample(sample_size, random_state=42) if len(df_pca) > sample_size else df_pca
    
    print(f"K-Means completed in {time.time() - start_time:.2f} seconds")

    # Generate clusters plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=df_pca_sample)
    plt.title('Clusters after PCA')
    plt.savefig(os.path.join(output_dir, 'pca_clusters.png'))
    plt.close()
    
    # Add variance explained information
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, n_components + 1))
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # Display explained variance percentages
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total variance explained: {sum(explained_variance):.2%}")
    
    return df_pca