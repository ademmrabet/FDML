import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_data(df):
    # 1. Supprimer les colonnes avec trop de valeurs manquantes (> 50%)
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # 2. Supprimer les colonnes identifiants ou non pertinentes
    df = df.drop(columns=['Index', 'Id'], errors='ignore')
    
    # 3. Gérer les colonnes à haute cardinalité
    # Identifier les colonnes textuelles longues qui peuvent causer des problèmes de mémoire
    high_cardinality_cols = ['Title', 'Description', 'Society', 'location']
    for col in high_cardinality_cols:
        if col in df.columns:
            # Convertir en valeurs catégorielles plus simples ou supprimer selon le cas
            if col in ['Title', 'Description']:
                # Ces colonnes sont probablement trop détaillées pour être utiles en one-hot encoding
                df = df.drop(columns=[col], errors='ignore')
            elif col in ['Society', 'location']:
                # Pour les colonnes comme Society et location, garder les valeurs les plus fréquentes
                value_counts = df[col].value_counts()
                top_values = value_counts[value_counts > 10].index  # Garder uniquement les valeurs qui apparaissent >10 fois
                df[col] = df[col].apply(lambda x: x if x in top_values else 'Other')

    # 4. Séparer les variables numériques et catégorielles
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # 5. Définir la variable cible
    target_col = 'Price (in rupees)'
    if target_col in num_cols:
        num_features = [col for col in num_cols if col != target_col]
        target = df[target_col]
    else:
        num_features = num_cols.tolist()
        target = None
        print("AVERTISSEMENT: Variable cible non trouvée dans les données!")

    # 6. Créer des pipelines pour le prétraitement
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 7. Créer le transformateur de colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )
    
    # 8. Appliquer les transformations
    features_processed = preprocessor.fit_transform(df)
    
    # 9. Créer un nouveau DataFrame avec les données transformées
    # Obtenir les noms des colonnes après one-hot encoding
    feature_names = num_features.copy()
    if len(cat_cols) > 0:
        ohe_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
        feature_names.extend(ohe_features)
    
    # Créer le DataFrame final
    df_processed = pd.DataFrame(features_processed, columns=feature_names)
    
    # Ajouter la variable cible
    if target is not None:
        df_processed[target_col] = target.values
    
    return df_processed