import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Données chargées avec succès depuis {path}. Nombre d'observations : {df.shape[0]}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

def save_processed_data(df, path):
    try:
        df.to_csv(path, index=False)
        print(f"Données traitées sauvegardées avec succès dans {path}.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données : {e}")
