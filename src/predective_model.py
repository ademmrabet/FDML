import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_predictive_models(df, output_dir):
    if 'SalePrice' not in df.columns:
        print("La colonne 'SalePrice' est manquante. La modélisation prédictive ne peut pas être effectuée.")
        return
    target_col = 'Price (in rupees)'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n\n[{name}] Résultats:")
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.2f}")

        # Visualisation
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{name} - Réel vs Prédit')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.tight_layout()

        filename = os.path.join(output_dir, f'{name}_prediction.png')
        plt.savefig(filename)
        plt.close()

        print(f"Graphique de performance sauvegardé sous: {filename}")
