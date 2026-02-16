"""
Script principal pour exécuter tout le pipeline
EDA + Prétraitement + Vectorisation
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / 'src'))

from load_data import load_all_numerical_data, load_all_text_data
from eda_num import comprehensive_eda_numerical
from eda_text import comprehensive_eda_text
from preprocess_num import preprocess_numerical_data
from preprocess_text import preprocess_text_data
from vectorize_text import vectorize_all_text_data

def main():
    """
    Pipeline complet du projet
    """
    print("\n" + "="*70)
    print(" "*10 + "🚀 FINANCIAL STRESS DETECTION PROJECT")
    print(" "*15 + "EDA + Preprocessing + Vectorization")
    print("="*70)
    
    # ========== PARTIE 1 : DONNÉES NUMÉRIQUES ==========
    print("\n\n" + "#"*70)
    print("#" + " "*22 + "PARTIE 1 : DONNÉES NUMÉRIQUES" + " "*19 + "#")
    print("#"*70)
    
    # 1.1 Chargement
    print("\n📂 Chargement des données numériques...")
    num_data = load_all_numerical_data()
    
    # 1.2 EDA
    print("\n🔍 Analyse exploratoire numérique...")
    stats_num = comprehensive_eda_numerical(num_data)
    
    # 1.3 Prétraitement
    print("\n🧹 Prétraitement numérique...")
    processed_num_data = preprocess_numerical_data(num_data, remove_outliers_flag=False)
    
    # ========== PARTIE 2 : DONNÉES TEXTUELLES ==========
    print("\n\n" + "#"*70)
    print("#" + " "*21 + "PARTIE 2 : DONNÉES TEXTUELLES" + " "*20 + "#")
    print("#"*70)
    
    # 2.1 Chargement
    print("\n📂 Chargement des données textuelles...")
    df_train, df_test = load_all_text_data()
    
    # 2.2 EDA
    print("\n🔍 Analyse exploratoire textuelle...")
    stats_text = comprehensive_eda_text(df_train, df_test)
    
    # 2.3 Prétraitement
    print("\n🧹 Prétraitement textuel...")
    df_train_clean, df_test_clean = preprocess_text_data(df_train, df_test)
    
    # 2.4 Vectorisation
    print("\n🎯 Vectorisation...")
    vectorize_all_text_data(df_train_clean, df_test_clean)
    
    # ========== RÉSUMÉ FINAL ==========
    print("\n\n" + "="*70)
    print(" "*20 + "✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*70)
    
    print("\n📊 RÉSUMÉ:")
    print(f"\n  Données numériques:")
    for year in num_data.keys():
        print(f"    - {year}: {processed_num_data[year].shape}")
    
    print(f"\n  Données textuelles:")
    print(f"    - Train: {df_train_clean.shape}")
    print(f"    - Test:  {df_test_clean.shape}")
    
    print(f"\n📁 Tous les résultats sont sauvegardés dans:")
    print(f"    - Figures:     outputs/figures/")
    print(f"    - Données:     outputs/processed_data/")
    print(f"    - Vecteurs:    outputs/vectorized_data/")
    
    print("\n" + "="*70)
    print("🎉 Vous pouvez maintenant passer à la modélisation!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()