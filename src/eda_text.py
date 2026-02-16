"""
Analyse Exploratoire des Données Textuelles
Statistiques de texte, visualisations, word clouds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from typing import Tuple, List
import config

# Télécharger les stopwords si nécessaire
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

# Configuration
sns.set_style(config.STYLE)
plt.rcParams['figure.dpi'] = config.DPI

def get_text_stats(df: pd.DataFrame, text_column: str = 'sentence') -> pd.DataFrame:
    """
    Obtenir les statistiques de base sur le texte
    
    Args:
        df: DataFrame
        text_column: Nom de la colonne contenant le texte
    
    Returns:
        DataFrame avec statistiques
    """
    df = df.copy()
    
    # Nombre de caractères
    df['char_count'] = df[text_column].str.len()
    
    # Nombre de mots
    df['word_count'] = df[text_column].str.split().str.len()
    
    # Nombre moyen de caractères par mot
    df['avg_word_length'] = df['char_count'] / df['word_count']
    
    stats = pd.DataFrame({
        'metric': ['Caractères', 'Mots', 'Longueur moy. mot'],
        'mean': [
            df['char_count'].mean(),
            df['word_count'].mean(),
            df['avg_word_length'].mean()
        ],
        'std': [
            df['char_count'].std(),
            df['word_count'].std(),
            df['avg_word_length'].std()
        ],
        'min': [
            df['char_count'].min(),
            df['word_count'].min(),
            df['avg_word_length'].min()
        ],
        'max': [
            df['char_count'].max(),
            df['word_count'].max(),
            df['avg_word_length'].max()
        ]
    })
    
    return stats, df

def analyze_label_distribution(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Analyser la distribution des labels (sentiment)
    
    Args:
        df_train: DataFrame d'entraînement
        df_test: DataFrame de test
    """
    print("\n" + "="*50)
    print("DISTRIBUTION DES LABELS")
    print("="*50)
    
    # Identifier la colonne de label
    label_col = None
    for col in ['label', 'sentiment', 'class']:
        if col in df_train.columns:
            label_col = col
            break
    
    if label_col is None:
        print("⚠️ Colonne de label non trouvée")
        return
    
    # Statistiques train
    train_counts = df_train[label_col].value_counts()
    train_pct = df_train[label_col].value_counts(normalize=True) * 100
    
    print(f"\n📊 TRAIN SET (n={len(df_train)})")
    for label, count in train_counts.items():
        print(f"  {label}: {count} ({train_pct[label]:.2f}%)")
    
    # Statistiques test
    test_counts = df_test[label_col].value_counts()
    test_pct = df_test[label_col].value_counts(normalize=True) * 100
    
    print(f"\n📊 TEST SET (n={len(df_test)})")
    for label, count in test_counts.items():
        print(f"  {label}: {count} ({test_pct[label]:.2f}%)")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train
    train_counts.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#95a5a6', '#2ecc71'])
    axes[0].set_title('Distribution des labels - TRAIN')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Nombre')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    # Test
    test_counts.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#95a5a6', '#2ecc71'])
    axes[1].set_title('Distribution des labels - TEST')
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Nombre')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'label_distribution_text.png', bbox_inches='tight')
    plt.show()

def plot_text_length_distribution(df: pd.DataFrame, dataset_name: str = 'train'):
    """
    Visualiser la distribution des longueurs de texte
    
    Args:
        df: DataFrame
        dataset_name: Nom du dataset
    """
    text_col = 'sentence' if 'sentence' in df.columns else df.columns[0]
    
    # Calculer les longueurs
    df['char_count'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution des caractères
    axes[0].hist(df['char_count'], bins=50, color='#3498db', edgecolor='black')
    axes[0].set_title(f'Distribution longueur (caractères) - {dataset_name}')
    axes[0].set_xlabel('Nombre de caractères')
    axes[0].set_ylabel('Fréquence')
    axes[0].axvline(df['char_count'].mean(), color='red', linestyle='--', 
                    label=f'Moyenne: {df["char_count"].mean():.1f}')
    axes[0].legend()
    
    # Distribution des mots
    axes[1].hist(df['word_count'], bins=50, color='#2ecc71', edgecolor='black')
    axes[1].set_title(f'Distribution longueur (mots) - {dataset_name}')
    axes[1].set_xlabel('Nombre de mots')
    axes[1].set_ylabel('Fréquence')
    axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                    label=f'Moyenne: {df["word_count"].mean():.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'text_length_distribution_{dataset_name}.png', 
                bbox_inches='tight')
    plt.show()

def get_most_common_words(text_series: pd.Series, n: int = 20, 
                          remove_stopwords: bool = True) -> List[Tuple[str, int]]:
    """
    Obtenir les mots les plus fréquents
    
    Args:
        text_series: Série de textes
        n: Nombre de mots à retourner
        remove_stopwords: Supprimer les stopwords
    
    Returns:
        Liste de tuples (mot, fréquence)
    """
    # Concaténer tous les textes
    all_text = ' '.join(text_series.astype(str))
    
    # Tokeniser
    words = re.findall(r'\b[a-z]+\b', all_text.lower())
    
    # Supprimer stopwords si demandé
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Compter
    word_counts = Counter(words)
    
    return word_counts.most_common(n)

def plot_word_frequency(df_train: pd.DataFrame, df_test: pd.DataFrame, top_n: int = 20):
    """
    Visualiser les mots les plus fréquents
    
    Args:
        df_train: DataFrame train
        df_test: DataFrame test
        top_n: Nombre de mots à afficher
    """
    text_col = 'sentence' if 'sentence' in df_train.columns else df_train.columns[0]
    
    # Mots les plus fréquents
    train_words = get_most_common_words(df_train[text_col], n=top_n)
    test_words = get_most_common_words(df_test[text_col], n=top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Train
    words_train, counts_train = zip(*train_words)
    axes[0].barh(range(len(words_train)), counts_train, color='#3498db')
    axes[0].set_yticks(range(len(words_train)))
    axes[0].set_yticklabels(words_train)
    axes[0].set_title(f'Top {top_n} mots - TRAIN')
    axes[0].set_xlabel('Fréquence')
    axes[0].invert_yaxis()
    
    # Test
    words_test, counts_test = zip(*test_words)
    axes[1].barh(range(len(words_test)), counts_test, color='#2ecc71')
    axes[1].set_yticks(range(len(words_test)))
    axes[1].set_yticklabels(words_test)
    axes[1].set_title(f'Top {top_n} mots - TEST')
    axes[1].set_xlabel('Fréquence')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'word_frequency.png', bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 TOP {top_n} MOTS LES PLUS FRÉQUENTS")
    print("\nTRAIN:")
    for word, count in train_words[:10]:
        print(f"  {word}: {count}")
    
    print("\nTEST:")
    for word, count in test_words[:10]:
        print(f"  {word}: {count}")

def create_wordcloud(text_series: pd.Series, title: str = "Word Cloud"):
    """
    Créer un word cloud
    
    Args:
        text_series: Série de textes
        title: Titre du graphique
    """
    # Concaténer tous les textes
    all_text = ' '.join(text_series.astype(str))
    
    # Nettoyer
    all_text = re.sub(r'[^a-zA-Z\s]', '', all_text.lower())
    
    # Créer le word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        stopwords=STOP_WORDS,
        colormap='viridis',
        max_words=100
    ).generate(all_text)
    
    # Afficher
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout(pad=0)
    
    # Sauvegarder
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(config.FIGURES_DIR / filename, bbox_inches='tight')
    plt.show()

def analyze_by_sentiment(df: pd.DataFrame, dataset_name: str = 'train'):
    """
    Analyser les textes par sentiment
    
    Args:
        df: DataFrame
        dataset_name: Nom du dataset
    """
    # Trouver la colonne de label
    label_col = None
    for col in ['label', 'sentiment', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("⚠️ Colonne de label non trouvée")
        return
    
    text_col = 'sentence' if 'sentence' in df.columns else df.columns[0]
    
    print(f"\n{'='*50}")
    print(f"ANALYSE PAR SENTIMENT - {dataset_name}")
    print(f"{'='*50}")
    
    # Pour chaque sentiment, créer un word cloud
    for sentiment in df[label_col].unique():
        df_sentiment = df[df[label_col] == sentiment]
        print(f"\n📊 Sentiment: {sentiment} (n={len(df_sentiment)})")
        
        # Statistiques de longueur
        avg_chars = df_sentiment[text_col].str.len().mean()
        avg_words = df_sentiment[text_col].str.split().str.len().mean()
        print(f"  Longueur moyenne: {avg_chars:.1f} caractères, {avg_words:.1f} mots")
        
        # Word cloud
        create_wordcloud(df_sentiment[text_col], 
                        title=f"Word Cloud - {sentiment} ({dataset_name})")

def comprehensive_eda_text(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    EDA complète pour les données textuelles
    
    Args:
        df_train: DataFrame train
        df_test: DataFrame test
    """
    print("\n" + "="*70)
    print(" "*15 + "📝 ANALYSE EXPLORATOIRE TEXTUELLE")
    print("="*70)
    
    text_col = 'sentence' if 'sentence' in df_train.columns else df_train.columns[0]
    
    # Statistiques de base
    print("\n📊 STATISTIQUES DE BASE\n")
    print(f"TRAIN: {df_train.shape}")
    print(f"TEST:  {df_test.shape}")
    
    stats_train, df_train_enriched = get_text_stats(df_train, text_col)
    stats_test, df_test_enriched = get_text_stats(df_test, text_col)
    
    print("\nTRAIN:")
    print(stats_train.to_string(index=False))
    
    print("\nTEST:")
    print(stats_test.to_string(index=False))
    
    # Distribution des labels
    analyze_label_distribution(df_train, df_test)
    
    # Distribution des longueurs
    plot_text_length_distribution(df_train_enriched, 'train')
    plot_text_length_distribution(df_test_enriched, 'test')
    
    # Fréquence des mots
    plot_word_frequency(df_train, df_test, top_n=20)
    
    # Word clouds globaux
    create_wordcloud(df_train[text_col], "Word Cloud - TRAIN (Global)")
    create_wordcloud(df_test[text_col], "Word Cloud - TEST (Global)")
    
    # Analyse par sentiment
    analyze_by_sentiment(df_train, 'train')
    analyze_by_sentiment(df_test, 'test')
    
    # Sauvegarder les statistiques
    stats_train.to_csv(config.PROCESSED_DIR / 'stats_text_train.csv', index=False)
    stats_test.to_csv(config.PROCESSED_DIR / 'stats_text_test.csv', index=False)
    
    print(f"\n💾 Statistiques sauvegardées")
    
    return stats_train, stats_test

if __name__ == "__main__":
    from load_data import load_all_text_data
    
    # Charger les données
    df_train, df_test = load_all_text_data()
    
    # Effectuer l'EDA
    stats_train, stats_test = comprehensive_eda_text(df_train, df_test)