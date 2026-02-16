"""
Prétraitement des données numériques
Gestion des valeurs manquantes, outliers, normalisation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Tuple
import config

def handle_missing_values(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Gérer les valeurs manquantes
    
    Args:
        df: DataFrame
        threshold: Seuil de suppression de colonnes (par défaut: config.MISSING_THRESHOLD)
    
    Returns:
        DataFrame nettoyé
    """
    if threshold is None:
        threshold = config.MISSING_THRESHOLD
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # 1. Supprimer les colonnes avec trop de valeurs manquantes
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"  🗑️  Suppression de {len(cols_to_drop)} colonnes (>{threshold*100}% manquant)")
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    # 2. Imputer les valeurs manquantes restantes avec la médiane
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'class']
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
    
    print(f"  ✅ Shape: {initial_shape} → {df_clean.shape}")
    
    return df_clean

def remove_outliers(df: pd.DataFrame, method: str = None, 
                   z_threshold: float = 3) -> pd.DataFrame:
    """
    Supprimer les outliers
    
    Args:
        df: DataFrame
        method: Méthode ('iqr' ou 'zscore')
        z_threshold: Seuil pour z-score
    
    Returns:
        DataFrame sans outliers
    """
    if method is None:
        method = config.OUTLIER_METHOD
    
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'class']
    
    if method == 'iqr':
        # Méthode IQR
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]
    
    elif method == 'zscore':
        # Méthode Z-score
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < z_threshold).all(axis=1)]
    
    removed = initial_len - len(df_clean)
    removed_pct = (removed / initial_len) * 100
    
    print(f"  🗑️  Outliers supprimés: {removed} ({removed_pct:.2f}%)")
    print(f"  ✅ Lignes restantes: {len(df_clean)}")
    
    return df_clean

def normalize_features(df: pd.DataFrame, method: str = None, 
                      exclude_cols: list = None) -> Tuple[pd.DataFrame, object]:
    """
    Normaliser les features
    
    Args:
        df: DataFrame
        method: Méthode ('standard', 'minmax', 'robust')
        exclude_cols: Colonnes à exclure
    
    Returns:
        Tuple (DataFrame normalisé, scaler)
    """
    if method is None:
        method = config.NORMALIZATION_METHOD
    
    if exclude_cols is None:
        exclude_cols = ['class']
    
    df_norm = df.copy()
    
    # Sélectionner les colonnes à normaliser
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Choisir le scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Normaliser
    df_norm[cols_to_normalize] = scaler.fit_transform(df_norm[cols_to_normalize])
    
    print(f"  ✅ Normalisation ({method}): {len(cols_to_normalize)} colonnes")
    
    return df_norm, scaler

def preprocess_numerical_data(num_data: Dict[str, pd.DataFrame], 
                              remove_outliers_flag: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Pipeline complet de prétraitement numérique
    
    Args:
        num_data: Dictionnaire {année: DataFrame}
        remove_outliers_flag: Supprimer les outliers ou non
    
    Returns:
        Dictionnaire {année: DataFrame nettoyé}
    """
    print("\n" + "="*70)
    print(" "*15 + "🧹 PRÉTRAITEMENT NUMÉRIQUE")
    print("="*70)
    
    processed_data = {}
    
    for year, df in num_data.items():
        print(f"\n{'='*50}")
        print(f"  {year.upper()}")
        print(f"{'='*50}")
        print(f"  Shape initiale: {df.shape}")
        
        # 1. Gérer les valeurs manquantes
        print("\n  1️⃣  Gestion des valeurs manquantes")
        df_clean = handle_missing_values(df)
        
        # 2. Supprimer les outliers (optionnel)
        if remove_outliers_flag:
            print("\n  2️⃣  Suppression des outliers")
            df_clean = remove_outliers(df_clean)
        else:
            print("\n  2️⃣  Outliers conservés (remove_outliers_flag=False)")
        
        # 3. Normaliser
        print("\n  3️⃣  Normalisation des features")
        df_norm, scaler = normalize_features(df_clean)
        
        processed_data[year] = df_norm
        
        # 4. Sauvegarder
        from load_data import save_to_csv
        save_to_csv(df_norm, f'processed_num_{year}')
        
        print(f"\n  ✅ {year} traité avec succès!\n")
    
    return processed_data

if __name__ == "__main__":
    from load_data import load_all_numerical_data
    
    # Charger les données
    num_data = load_all_numerical_data()
    
    # Prétraiter
    processed_data = preprocess_numerical_data(num_data, remove_outliers_flag=False)
    
    print("\n" + "="*70)
    print("✅ PRÉTRAITEMENT NUMÉRIQUE TERMINÉ")
    print("="*70)