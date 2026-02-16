

import pandas as pd
from scipy.io import arff
from pathlib import Path
from typing import Dict, Tuple
import config

def load_arff_file(filepath: Path) -> pd.DataFrame:
    """
    Charge un fichier ARFF et le convertit en DataFrame
    
    Args:
        filepath: Chemin vers le fichier .arff
    
    Returns:
        DataFrame pandas
    """
    try:
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        
        # Convertir les bytes en string pour la colonne target
        if 'class' in df.columns:
            df['class'] = df['class'].str.decode('utf-8')
        
        print(f"✅ Chargé : {filepath.name} - Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {filepath}: {e}")
        return None

def load_all_numerical_data() -> Dict[str, pd.DataFrame]:
    """
    Charge tous les fichiers numériques (ARFF)
    
    Returns:
        Dictionnaire {année: DataFrame}
    """
    numerical_data = {}
    
    for year, filepath in config.NUM_FILES.items():
        df = load_arff_file(filepath)
        if df is not None:
            numerical_data[year] = df
    
    print(f"\n📊 Total fichiers numériques chargés : {len(numerical_data)}")
    return numerical_data

def load_parquet_file(filepath: Path) -> pd.DataFrame:
    """
    Charge un fichier Parquet
    
    Args:
        filepath: Chemin vers le fichier .parquet
    
    Returns:
        DataFrame pandas
    """
    try:
        df = pd.read_parquet(filepath)
        print(f"✅ Chargé : {filepath.name} - Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {filepath}: {e}")
        return None

def load_all_text_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données textuelles (train et test)
    
    Returns:
        Tuple (df_train, df_test)
    """
    df_train = load_parquet_file(config.TEXT_FILES["train"])
    df_test = load_parquet_file(config.TEXT_FILES["test"])
    
    print(f"\n📝 Données textuelles chargées")
    return df_train, df_test

def save_to_csv(df: pd.DataFrame, filename: str, directory: Path = None):
    """
    Sauvegarde un DataFrame en CSV
    
    Args:
        df: DataFrame à sauvegarder
        filename: Nom du fichier (sans extension)
        directory: Dossier de destination (par défaut: PROCESSED_DIR)
    """
    if directory is None:
        directory = config.PROCESSED_DIR
    
    filepath = directory / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"💾 Sauvegardé : {filepath}")

if __name__ == "__main__":
    # Test du module
    print("=== TEST DE CHARGEMENT ===\n")
    
    # Charger données numériques
    num_data = load_all_numerical_data()
    
    # Charger données textuelles
    train_df, test_df = load_all_text_data()
    
    # Afficher aperçu
    if num_data:
        print("\n=== APERÇU DONNÉES NUMÉRIQUES (1year) ===")
        print(num_data["1year"].head())
        print(f"\nColonnes : {num_data['1year'].columns.tolist()[:10]}...")
    
    if train_df is not None:
        print("\n=== APERÇU DONNÉES TEXTUELLES (train) ===")
        print(train_df.head())
        print(f"\nColonnes : {train_df.columns.tolist()}")