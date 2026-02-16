"""
Prétraitement des données textuelles
Nettoyage, tokenisation, lemmatisation
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import config

# Télécharger les ressources NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Nettoyer un texte
    
    Args:
        text: Texte à nettoyer
    
    Returns:
        Texte nettoyé
    """
    if pd.isna(text):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Supprimer les mentions @ et hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    
    # Garder uniquement les lettres et espaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text: str) -> list:
    """
    Tokeniser un texte
    
    Args:
        text: Texte à tokeniser
    
    Returns:
        Liste de tokens
    """
    if not text:
        return []
    
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens: list) -> list:
    """
    Supprimer les stopwords
    
    Args:
        tokens: Liste de tokens
    
    Returns:
        Liste de tokens sans stopwords
    """
    return [token for token in tokens if token not in STOP_WORDS and len(token) > config.MIN_WORD_LENGTH]

def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatiser les tokens
    
    Args:
        tokens: Liste de tokens
    
    Returns:
        Liste de tokens lemmatisés
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text_pipeline(text: str) -> str:
    """
    Pipeline complet de prétraitement de texte
    
    Args:
        text: Texte brut
    
    Returns:
        Texte prétraité (string)
    """
    # 1. Nettoyer
    text = clean_text(text)
    
    # 2. Tokeniser
    tokens = tokenize_text(text)
    
    # 3. Supprimer stopwords
    tokens = remove_stopwords(tokens)
    
    # 4. Lemmatiser
    tokens = lemmatize_tokens(tokens)
    
    # 5. Rejoindre
    return ' '.join(tokens)

def preprocess_text_data(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                        text_column: str = 'sentence') -> tuple:
    """
    Prétraiter les données textuelles
    
    Args:
        df_train: DataFrame train
        df_test: DataFrame test
        text_column: Nom de la colonne texte
    
    Returns:
        Tuple (df_train_clean, df_test_clean)
    """
    print("\n" + "="*70)
    print(" "*15 + "🧹 PRÉTRAITEMENT TEXTUEL")
    print("="*70)
    
    # Copier les DataFrames
    df_train_clean = df_train.copy()
    df_test_clean = df_test.copy()
    
    print(f"\n📊 Shape initiale:")
    print(f"  TRAIN: {df_train_clean.shape}")
    print(f"  TEST:  {df_test_clean.shape}")
    
    # Vérifier la colonne texte
    if text_column not in df_train_clean.columns:
        print(f"⚠️ Colonne '{text_column}' non trouvée")
        print(f"   Colonnes disponibles: {df_train_clean.columns.tolist()}")
        # Prendre la première colonne par défaut
        text_column = df_train_clean.columns[0]
        print(f"   Utilisation de: {text_column}")
    
    # Prétraiter TRAIN
    print(f"\n🔄 Prétraitement TRAIN...")
    df_train_clean['text_clean'] = df_train_clean[text_column].apply(preprocess_text_pipeline)
    
    # Supprimer les lignes vides après nettoyage
    initial_len = len(df_train_clean)
    df_train_clean = df_train_clean[df_train_clean['text_clean'].str.len() > 0]
    removed = initial_len - len(df_train_clean)
    print(f"  ✅ Lignes vides supprimées: {removed}")
    print(f"  ✅ Shape finale: {df_train_clean.shape}")
    
    # Prétraiter TEST
    print(f"\n🔄 Prétraitement TEST...")
    df_test_clean['text_clean'] = df_test_clean[text_column].apply(preprocess_text_pipeline)
    
    initial_len = len(df_test_clean)
    df_test_clean = df_test_clean[df_test_clean['text_clean'].str.len() > 0]
    removed = initial_len - len(df_test_clean)
    print(f"  ✅ Lignes vides supprimées: {removed}")
    print(f"  ✅ Shape finale: {df_test_clean.shape}")
    
    # Afficher des exemples
    print(f"\n📝 EXEMPLES DE TRANSFORMATION:")
    print(f"\n{'='*70}")
    for i in range(min(3, len(df_train_clean))):
        print(f"\nEXEMPLE {i+1}:")
        print(f"  Original: {df_train_clean.iloc[i][text_column]}")
        print(f"  Nettoyé:  {df_train_clean.iloc[i]['text_clean']}")
        print("-"*70)
    
    # Sauvegarder
    from load_data import save_to_csv
    save_to_csv(df_train_clean, 'processed_text_train')
    save_to_csv(df_test_clean, 'processed_text_test')
    
    print(f"\n💾 Données sauvegardées dans {config.PROCESSED_DIR}")
    
    return df_train_clean, df_test_clean

if __name__ == "__main__":
    from load_data import load_all_text_data
    
    # Charger les données
    df_train, df_test = load_all_text_data()
    
    # Prétraiter
    df_train_clean, df_test_clean = preprocess_text_data(df_train, df_test)
    
    print("\n" + "="*70)
    print("✅ PRÉTRAITEMENT TEXTUEL TERMINÉ")
    print("="*70)

