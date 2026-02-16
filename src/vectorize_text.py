"""
Vectorisation des données textuelles
TF-IDF, embeddings, sauvegarde des vecteurs
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import save_npz
import pickle
import config

def create_tfidf_vectors(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                        text_column: str = 'text_clean',
                        max_features: int = None,
                        ngram_range: tuple = None) -> tuple:
    """
    Créer des vecteurs TF-IDF
    
    Args:
        df_train: DataFrame train
        df_test: DataFrame test
        text_column: Colonne contenant le texte nettoyé
        max_features: Nombre max de features
        ngram_range: Range des n-grammes
    
    Returns:
        Tuple (X_train_tfidf, X_test_tfidf, vectorizer, feature_names)
    """
    if max_features is None:
        max_features = config.MAX_FEATURES_TFIDF
    
    if ngram_range is None:
        ngram_range = config.NGRAM_RANGE
    
    print("\n" + "="*70)
    print(" "*20 + "🎯 VECTORISATION TF-IDF")
    print("="*70)
    
    print(f"\n⚙️  Paramètres:")
    print(f"  max_features: {max_features}")
    print(f"  ngram_range:  {ngram_range}")
    
    # Créer le vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,  # Ignorer les termes qui apparaissent dans moins de 2 documents
        max_df=0.95,  # Ignorer les termes qui apparaissent dans plus de 95% des documents
        sublinear_tf=True  # Appliquer log scaling
    )
    
    # Fit sur train, transform sur train et test
    print(f"\n🔄 Vectorisation en cours...")
    X_train_tfidf = vectorizer.fit_transform(df_train[text_column])
    X_test_tfidf = vectorizer.transform(df_test[text_column])
    
    # Récupérer les noms de features
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n✅ Vectorisation terminée!")
    print(f"  Shape TRAIN: {X_train_tfidf.shape}")
    print(f"  Shape TEST:  {X_test_tfidf.shape}")
    print(f"  Vocabulaire: {len(feature_names)} termes")
    
    # Top features
    print(f"\n📊 Top 20 features (TF-IDF moyen):")
    tfidf_means = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
    top_indices = tfidf_means.argsort()[-20:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:20s} (score: {tfidf_means[idx]:.4f})")
    
    return X_train_tfidf, X_test_tfidf, vectorizer, feature_names

def create_bow_vectors(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      text_column: str = 'text_clean',
                      max_features: int = None) -> tuple:
    """
    Créer des vecteurs Bag of Words
    
    Args:
        df_train: DataFrame train
        df_test: DataFrame test
        text_column: Colonne texte
        max_features: Nombre max de features
    
    Returns:
        Tuple (X_train_bow, X_test_bow, vectorizer, feature_names)
    """
    if max_features is None:
        max_features = config.MAX_FEATURES_TFIDF
    
    print("\n" + "="*70)
    print(" "*20 + "📊 VECTORISATION BAG OF WORDS")
    print("="*70)
    
    # Créer le vectorizer
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95
    )
    
    # Fit et transform
    print(f"\n🔄 Vectorisation en cours...")
    X_train_bow = vectorizer.fit_transform(df_train[text_column])
    X_test_bow = vectorizer.transform(df_test[text_column])
    
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n✅ Vectorisation terminée!")
    print(f"  Shape TRAIN: {X_train_bow.shape}")
    print(f"  Shape TEST:  {X_test_bow.shape}")
    
    return X_train_bow, X_test_bow, vectorizer, feature_names

def save_vectors_and_vectorizer(X_train, X_test, vectorizer, 
                               method: str = 'tfidf'):
    """
    Sauvegarder les vecteurs et le vectorizer
    
    Args:
        X_train: Matrice sparse train
        X_test: Matrice sparse test
        vectorizer: Vectorizer fitted
        method: Nom de la méthode ('tfidf' ou 'bow')
    """
    print(f"\n💾 Sauvegarde des vecteurs ({method})...")
    
    # Sauvegarder les matrices sparse
    save_npz(config.VECTORIZED_DIR / f'X_train_{method}.npz', X_train)
    save_npz(config.VECTORIZED_DIR / f'X_test_{method}.npz', X_test)
    
    # Sauvegarder le vectorizer
    with open(config.VECTORIZED_DIR / f'vectorizer_{method}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"  ✅ Matrices sauvegardées: X_train_{method}.npz, X_test_{method}.npz")
    print(f"  ✅ Vectorizer sauvegardé: vectorizer_{method}.pkl")

def create_feature_importance_df(vectorizer, X_train, feature_names, top_n: int = 50):
    """
    Créer un DataFrame avec l'importance des features
    
    Args:
        vectorizer: Vectorizer
        X_train: Matrice train
        feature_names: Noms des features
        top_n: Nombre de top features
    
    Returns:
        DataFrame
    """
    # Calculer le score TF-IDF moyen pour chaque terme
    tfidf_means = np.asarray(X_train.mean(axis=0)).flatten()
    
    # Créer le DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'tfidf_mean': tfidf_means
    })
    
    df_importance = df_importance.sort_values('tfidf_mean', ascending=False)
    
    # Sauvegarder
    df_importance.head(top_n).to_csv(
        config.PROCESSED_DIR / 'feature_importance_tfidf.csv', 
        index=False
    )
    
    return df_importance

def vectorize_all_text_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Vectoriser toutes les données textuelles
    
    Args:
        df_train: DataFrame train nettoyé
        df_test: DataFrame test nettoyé
    """
    print("\n" + "="*70)
    print(" "*15 + "🚀 PIPELINE DE VECTORISATION COMPLET")
    print("="*70)
    
    # 1. TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer, tfidf_features = create_tfidf_vectors(
        df_train, df_test
    )
    
    save_vectors_and_vectorizer(X_train_tfidf, X_test_tfidf, tfidf_vectorizer, 'tfidf')
    
    # Feature importance
    df_importance = create_feature_importance_df(
        tfidf_vectorizer, X_train_tfidf, tfidf_features
    )
    
    print(f"\n📊 Top 10 features les plus importantes:")
    print(df_importance.head(10).to_string(index=False))
    
    # 2. Bag of Words (optionnel)
    print("\n" + "-"*70)
    X_train_bow, X_test_bow, bow_vectorizer, bow_features = create_bow_vectors(
        df_train, df_test
    )
    
    save_vectors_and_vectorizer(X_train_bow, X_test_bow, bow_vectorizer, 'bow')
    
    print("\n" + "="*70)
    print("✅ VECTORISATION TERMINÉE")
    print("="*70)
    print(f"\n📁 Fichiers sauvegardés dans: {config.VECTORIZED_DIR}")

if __name__ == "__main__":
    from load_data import load_all_text_data
    from preprocess_text import preprocess_text_data
    
    # Charger et prétraiter
    df_train, df_test = load_all_text_data()
    df_train_clean, df_test_clean = preprocess_text_data(df_train, df_test)
    
    # Vectoriser
    vectorize_all_text_data(df_train_clean, df_test_clean)