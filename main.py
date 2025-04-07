import os
import sys
import pandas as pd
import time

# Add src folder to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

from data_exploration import explore_data
from data_preparation import prepare_data
from descriptive_model import run_descriptive_models
from predective_model import run_predictive_models
from utils import load_data, save_processed_data

# === CONFIGURATION ===
DATA_PATH = os.path.join('data', 'house_prices.csv')
PROCESSED_PATH = os.path.join('data', 'processed_data.csv')
VISUALIZATION_DIR = 'visualizations'

def main():
    # Create visualization directory if it doesn't exist
    if not os.path.exists(VISUALIZATION_DIR):
        os.makedirs(VISUALIZATION_DIR)

    print("\n[1] Chargement des données...")
    df = load_data(DATA_PATH)

    # Check if we should run data exploration
    exploration_flag = os.path.join(VISUALIZATION_DIR, '.exploration_done')
    if not os.path.exists(exploration_flag):
        print("\n[2] Exploration des données...")
        explore_data(df, VISUALIZATION_DIR)
        # Create flag file to indicate exploration is done
        with open(exploration_flag, 'w') as f:
            f.write('done')
    else:
        print("\n[2] Exploration des données... (déjà effectuée)")

    # Check if processed data already exists
    if os.path.exists(PROCESSED_PATH):
        print(f"\n[3] Chargement des données prétraitées depuis {PROCESSED_PATH}...")
        start_time = time.time()
        df_processed = pd.read_csv(PROCESSED_PATH)
        print(f"Données prétraitées chargées en {time.time() - start_time:.2f} secondes")
    else:
        print("\n[3] Préparation des données...")
        start_time = time.time()
        df_processed = prepare_data(df)
        save_processed_data(df_processed, PROCESSED_PATH)
        print(f"Préparation des données effectuée en {time.time() - start_time:.2f} secondes")

    print("\n[4] Modélisation descriptive (PCA, KMeans)...")
    descriptive_flag = os.path.join(VISUALIZATION_DIR, '.descriptive_done')
    if not os.path.exists(descriptive_flag):
        start_time = time.time()
        run_descriptive_models(df_processed, VISUALIZATION_DIR)
        print(f"Modélisation descriptive effectuée en {time.time() - start_time:.2f} secondes")
        # Create flag file to indicate descriptive modeling is done
        with open(descriptive_flag, 'w') as f:
            f.write('done')
    else:
        print("Modélisation descriptive... (déjà effectuée)")

    print("\n[5] Modélisation prédictive (Régression linéaire, Random Forest)...")
    predictive_flag = os.path.join(VISUALIZATION_DIR, '.predictive_done')
    if not os.path.exists(predictive_flag):
        start_time = time.time()
        run_predictive_models(df_processed, VISUALIZATION_DIR)
        print(f"Modélisation prédictive effectuée en {time.time() - start_time:.2f} secondes")
        # Create flag file to indicate predictive modeling is done
        with open(predictive_flag, 'w') as f:
            f.write('done')
    else:
        print("Modélisation prédictive... (déjà effectuée)")

    print("\n[✔] Pipeline terminé avec succès.")
    print(f"Les résultats visuels sont dans le dossier : {VISUALIZATION_DIR}")

if __name__ == '__main__':
    main()