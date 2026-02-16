"""
Analyse Exploratoire des Données Numériques
Statistiques, visualisations, détection d'anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import config

# Configuration du style
sns.set_style(config.STYLE)
plt.rcParams['figure.dpi'] = config.DPI

def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtenir les statistiques descriptives de base
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        DataFrame avec statistiques
    """
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True)
    })
    
    return stats

def analyze_target_distribution(df: pd.DataFrame, year: str):
    """
    Analyser la distribution de la variable cible (bankruptcy)
    
    Args:
        df: DataFrame
        year: Année du dataset
    """
    if 'class' not in df.columns:
        print("⚠️ Colonne 'class' non trouvée")
        return
    
    # Compter les valeurs
    target_counts = df['class'].value_counts()
    target_pct = df['class'].value_counts(normalize=True) * 100
    
    print(f"\n{'='*50}")
    print(f"DISTRIBUTION CIBLE - {year}")
    print(f"{'='*50}")
    print(f"\nNombre d'observations :")
    for label, count in target_counts.items():
        print(f"  {label}: {count} ({target_pct[label]:.2f}%)")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Barplot
    target_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title(f'Distribution des classes - {year}')
    axes[0].set_xlabel('Classe')
    axes[0].set_ylabel('Nombre')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Pie chart
    axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title(f'Proportion - {year}')
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'target_distribution_{year}.png', bbox_inches='tight')
    plt.show()

def plot_missing_values(df: pd.DataFrame, year: str):
    """
    Visualiser les valeurs manquantes
    
    Args:
        df: DataFrame
        year: Année du dataset
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print(f"✅ Aucune valeur manquante dans {year}")
        return
    
    print(f"\n{'='*50}")
    print(f"VALEURS MANQUANTES - {year}")
    print(f"{'='*50}")
    print(f"\nTop 10 colonnes avec valeurs manquantes :")
    print(missing.head(10))
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    missing.head(20).plot(kind='barh', color='#e74c3c')
    plt.title(f'Top 20 colonnes avec valeurs manquantes - {year}')
    plt.xlabel('Nombre de valeurs manquantes')
    plt.ylabel('Colonnes')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'missing_values_{year}.png', bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, year: str, top_n: int = 30):
    """
    Visualiser la matrice de corrélation
    
    Args:
        df: DataFrame
        year: Année
        top_n: Nombre de features à afficher
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclure la colonne cible si elle existe
    numeric_cols = [col for col in numeric_cols if col != 'class']
    
    if len(numeric_cols) == 0:
        print("⚠️ Aucune colonne numérique trouvée")
        return
    
    # Limiter au top_n features
    if len(numeric_cols) > top_n:
        # Calculer la variance et garder les top_n
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(top_n).index.tolist()
    
    # Calculer la corrélation
    corr_matrix = df[numeric_cols].corr()
    
    # Visualisation
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot=False)
    plt.title(f'Matrice de corrélation (Top {top_n} features) - {year}')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'correlation_matrix_{year}.png', bbox_inches='tight')
    plt.show()

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Détecter les outliers avec la méthode IQR
    
    Args:
        df: DataFrame
        column: Nom de la colonne
    
    Returns:
        Série booléenne (True = outlier)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def analyze_outliers(df: pd.DataFrame, year: str, sample_cols: int = 5):
    """
    Analyser les outliers sur quelques colonnes
    
    Args:
        df: DataFrame
        year: Année
        sample_cols: Nombre de colonnes à analyser
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'class']
    
    if len(numeric_cols) == 0:
        return
    
    # Prendre un échantillon de colonnes
    sample_cols = min(sample_cols, len(numeric_cols))
    cols_to_analyze = np.random.choice(numeric_cols, sample_cols, replace=False)
    
    print(f"\n{'='*50}")
    print(f"ANALYSE OUTLIERS (échantillon) - {year}")
    print(f"{'='*50}")
    
    # Boxplots
    fig, axes = plt.subplots(1, sample_cols, figsize=(15, 4))
    if sample_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(cols_to_analyze):
        outliers = detect_outliers_iqr(df, col)
        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        print(f"\n{col}:")
        print(f"  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
        
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'{col}\n({outlier_count} outliers)')
        axes[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'outliers_analysis_{year}.png', bbox_inches='tight')
    plt.show()

def plot_feature_distributions(df: pd.DataFrame, year: str, n_features: int = 6):
    """
    Visualiser la distribution de quelques features
    
    Args:
        df: DataFrame
        year: Année
        n_features: Nombre de features à visualiser
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'class']
    
    if len(numeric_cols) == 0:
        return
    
    # Sélectionner aléatoirement n_features
    n_features = min(n_features, len(numeric_cols))
    selected_features = np.random.choice(numeric_cols, n_features, replace=False)
    
    # Créer les subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            df[col].hist(bins=50, ax=axes[i], color='#3498db', edgecolor='black')
            axes[i].set_title(f'Distribution de {col}')
            axes[i].set_xlabel('Valeur')
            axes[i].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'feature_distributions_{year}.png', bbox_inches='tight')
    plt.show()

def comprehensive_eda_numerical(num_data: Dict[str, pd.DataFrame]):
    """
    EDA complète pour toutes les données numériques
    
    Args:
        num_data: Dictionnaire {année: DataFrame}
    """
    print("\n" + "="*70)
    print(" "*15 + "🔍 ANALYSE EXPLORATOIRE NUMÉRIQUE")
    print("="*70)
    
    all_stats = {}
    
    for year, df in num_data.items():
        print(f"\n\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"  ANNÉE: {year.upper()}")
        print(f"{'#'*70}")
        print(f"{'#'*70}")
        
        # Statistiques de base
        print(f"\n📊 STATISTIQUES DE BASE")
        print(f"  Shape: {df.shape}")
        print(f"  Mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        stats = get_basic_stats(df)
        all_stats[year] = stats
        
        print(f"\n  Top 10 colonnes avec le plus de valeurs manquantes:")
        print(stats.nlargest(10, 'missing')[['missing', 'missing_pct']])
        
        # Analyse cible
        analyze_target_distribution(df, year)
        
        # Valeurs manquantes
        plot_missing_values(df, year)
        
        # Corrélations
        plot_correlation_matrix(df, year, top_n=30)
        
        # Outliers
        analyze_outliers(df, year, sample_cols=5)
        
        # Distributions
        plot_feature_distributions(df, year, n_features=6)
        
        # Sauvegarder les stats
        stats.to_csv(config.PROCESSED_DIR / f'stats_{year}.csv')
        print(f"\n💾 Statistiques sauvegardées : stats_{year}.csv")
    
    return all_stats

if __name__ == "__main__":
    from load_data import load_all_numerical_data
    
    # Charger les données
    num_data = load_all_numerical_data()
    
    # Effectuer l'EDA
    stats = comprehensive_eda_numerical(num_data)