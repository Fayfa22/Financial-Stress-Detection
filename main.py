"""
Pipeline complet :
EDA → Preprocessing → Vectorisation → Modélisation MLflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from load_data        import load_all_numerical_data, load_all_text_data
from preprocess_num   import preprocess_numerical_data
from preprocess_text  import preprocess_text_data
from vectorize_text   import vectorize_all_text_data
from train_num_model  import train_all_numerical_models
from train_text_model import train_all_text_models

def main():
    print("\n" + "="*70)
    print(" "*10 + "🚀 FINANCIAL STRESS DETECTION PROJECT")
    print("="*70)

    # ── Numérique ─────────────────────────────────────────────────────
    print("\n\n" + "#"*70)
    print("  PARTIE 1 : DONNÉES NUMÉRIQUES")
    print("#"*70)

    num_data      = load_all_numerical_data()
    processed_num = preprocess_numerical_data(num_data)
    train_all_numerical_models(processed_num)

    # ── Textuel ───────────────────────────────────────────────────────
    print("\n\n" + "#"*70)
    print("  PARTIE 2 : DONNÉES TEXTUELLES")
    print("#"*70)

    df_train, df_test             = load_all_text_data()
    df_train_clean, df_test_clean = preprocess_text_data(df_train, df_test)
    vectorize_all_text_data(df_train_clean, df_test_clean)
    train_all_text_models()

    # ── Fin ───────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ✅ PIPELINE TERMINÉ")
    print("="*70)
    print("\n  📁 Modèles     → models/")
    print("  📊 MLflow UI   → mlflow ui  (puis http://127.0.0.1:5000)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()