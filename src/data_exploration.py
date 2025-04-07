import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def explore_data(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Aperçu général
    df.describe(include='all').to_csv(os.path.join(save_dir, 'description_generale.csv'))

    # Matrice de corrélation
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"))
    plt.close()

    # Distribution des prix
    if 'SalePrice' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[''], kde=True, bins=30)
        plt.title("Distribution des prix de vente")
        plt.xlabel("Prix")
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_prix.png"))
        plt.close()

    # Prix vs surface habitable
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
        plt.title("Prix vs surface habitable")
        plt.xlabel("Surface habitable (GrLivArea)")
        plt.ylabel("Prix de vente")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prix_vs_surface.png"))
        plt.close()

    # Prix moyen par quartier
    if 'Neighborhood' in df.columns and 'SalePrice' in df.columns:
        plt.figure(figsize=(12, 6))
        grouped = df.groupby('Neighborhood')['SalePrice'].mean().sort_values()
        grouped.plot(kind='bar')
        plt.title("Prix moyen par quartier")
        plt.xlabel("Quartier")
        plt.ylabel("Prix moyen")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prix_par_quartier.png"))
        plt.close()