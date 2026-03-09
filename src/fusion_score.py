"""
Fusion des scores numériques et textuels
Score de stress financier unifié
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import config


# ══════════════════════════════════════════════════════════
# CALCUL DES SCORES INDIVIDUELS
# ══════════════════════════════════════════════════════════

def compute_numerical_stress_score(X_input: pd.DataFrame,
                                    model_path: str = None) -> np.ndarray:
    """
    Calculer le score de stress numérique
    
    Args:
        X_input: DataFrame avec features numériques
        model_path: Chemin vers le modèle (optionnel)
    
    Returns:
        Probabilités de faillite [0, 1]
    """
    if model_path is None:
        model_path = config.MODELS_DIR / "best_num_model.pkl"
    
    model = joblib.load(model_path)
    X_input = X_input.fillna(X_input.median())
    proba = model.predict_proba(X_input)[:, 1]
    
    return proba


def compute_textual_stress_score(texts: list,
                                  model_path: str = None,
                                  vectorizer_path: str = None) -> np.ndarray:
    """
    Calculer le score de stress textuel
    
    Args:
        texts: Liste de textes
        model_path: Chemin vers le modèle textuel
        vectorizer_path: Chemin vers le vectorizer
    
    Returns:
        Score de stress textuel [0, 1] (probabilité de sentiment négatif)
    """
    if model_path is None:
        model_path = config.MODELS_DIR / "best_text_model.pkl"
    if vectorizer_path is None:
        vectorizer_path = config.MODELS_DIR / "tfidf_vectorizer.pkl"
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    le = joblib.load(config.MODELS_DIR / "label_encoder_text.pkl")
    
    # Vectoriser
    X_text = vectorizer.transform(texts)
    proba = model.predict_proba(X_text)
    
    # Trouver l'index de la classe négative
    classes = list(le.classes_)
    neg_keywords = ['negative', 'neg', '0', 'bearish']
    neg_idx = 0
    for i, c in enumerate(classes):
        if any(k in str(c).lower() for k in neg_keywords):
            neg_idx = i
            break
    
    stress_score = proba[:, neg_idx]
    return stress_score


# ══════════════════════════════════════════════════════════
# FUSION DES SCORES
# ══════════════════════════════════════════════════════════

def fuse_scores(numerical_scores: np.ndarray,
                textual_scores: np.ndarray,
                weight_num: float = None,
                weight_text: float = None) -> np.ndarray:
    """
    Fusionner les scores numériques et textuels
    
    Args:
        numerical_scores: Scores numériques [0, 1]
        textual_scores: Scores textuels [0, 1]
        weight_num: Poids du score numérique
        weight_text: Poids du score textuel
    
    Returns:
        Score de stress fusionné [0, 1]
    """
    if weight_num is None:
        weight_num = config.WEIGHT_NUMERICAL
    if weight_text is None:
        weight_text = config.WEIGHT_TEXTUAL
    
    # Normaliser les poids
    total = weight_num + weight_text
    weight_num /= total
    weight_text /= total
    
    fused = (weight_num * numerical_scores) + (weight_text * textual_scores)
    return np.clip(fused, 0, 1)


# ══════════════════════════════════════════════════════════
# INTERPRÉTATION DES SCORES
# ══════════════════════════════════════════════════════════

def interpret_stress_score(score: float) -> dict:
    """
    Interpréter un score de stress financier
    
    Args:
        score: Score de stress [0, 1]
    
    Returns:
        Dictionnaire avec niveau, emoji, message
    """
    if score < 0.2:
        return {
            'level': 'FAIBLE',
            'emoji': '🟢',
            'message': 'Situation financière saine.'
        }
    elif score < 0.4:
        return {
            'level': 'MODÉRÉ',
            'emoji': '🟡',
            'message': 'Légères tensions. Surveillance recommandée.'
        }
    elif score < 0.6:
        return {
            'level': 'ÉLEVÉ',
            'emoji': '🟠',
            'message': 'Stress significatif. Action corrective conseillée.'
        }
    elif score < 0.8:
        return {
            'level': 'CRITIQUE',
            'emoji': '🔴',
            'message': 'Stress critique. Intervention urgente.'
        }
    else:
        return {
            'level': 'EXTRÊME',
            'emoji': '⛔',
            'message': 'Risque de faillite très élevé.'
        }


# ══════════════════════════════════════════════════════════
# ANALYSE DE DIVERGENCE
# ══════════════════════════════════════════════════════════

def analyze_divergence(numerical_scores: np.ndarray,
                        textual_scores: np.ndarray) -> pd.DataFrame:
    """
    Analyser la divergence entre scores numériques et textuels
    
    Args:
        numerical_scores: Scores numériques
        textual_scores: Scores textuels
    
    Returns:
        DataFrame avec analyse complète
    """
    divergence = np.abs(numerical_scores - textual_scores)
    
    df = pd.DataFrame({
        'numerical_score': numerical_scores,
        'textual_score': textual_scores,
        'fused_score': fuse_scores(numerical_scores, textual_scores),
        'divergence': divergence
    })
    
    df['divergence_level'] = pd.cut(
        df['divergence'],
        bins=[0, 0.2, 0.4, 1.0],
        labels=['Faible', 'Modérée', 'Forte']
    )
    
    print("\n📊 ANALYSE DE DIVERGENCE")
    print(f"  Divergence moyenne : {divergence.mean():.4f}")
    print(f"  Divergence max     : {divergence.max():.4f}")
    print(f"\n  Distribution :\n{df['divergence_level'].value_counts().to_string()}")
    
    return df


# ══════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════

def plot_score_distributions(df: pd.DataFrame):
    """
    Visualiser la distribution des 3 scores
    
    Args:
        df: DataFrame retourné par analyze_divergence()
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {
        'numerical_score': '#3498db',
        'textual_score': '#2ecc71',
        'fused_score': '#9b59b6'
    }
    
    titles = {
        'numerical_score': 'Score Numérique',
        'textual_score': 'Score Textuel',
        'fused_score': 'Score Fusionné'
    }
    
    for ax, (col, color) in zip(axes, colors.items()):
        ax.hist(df[col], bins=30, color=color, edgecolor='black', alpha=0.8)
        ax.axvline(df[col].mean(), color='red', linestyle='--',
                   label=f'μ={df[col].mean():.3f}')
        ax.set_title(titles[col])
        ax.set_xlabel('Score')
        ax.set_ylabel('Fréquence')
        ax.legend()
    
    plt.tight_layout()
    path = config.FIGURES_DIR / 'score_distributions.png'
    plt.savefig(path, dpi=config.DPI, bbox_inches='tight')
    plt.show()
    print(f"💾 Graphique sauvegardé : {path}")


if __name__ == "__main__":
    # Test rapide
    print("✅ Module fusion_score chargé avec succès")
    
    # Test des fonctions
    num_scores = np.random.uniform(0, 1, 100)
    text_scores = np.random.uniform(0, 1, 100)
    
    df = analyze_divergence(num_scores, text_scores)
    plot_score_distributions(df)
    
    # Test d'interprétation
    print("\n📊 TEST D'INTERPRÉTATION DES SCORES")
    for score in [0.1, 0.35, 0.55, 0.75, 0.92]:
        result = interpret_stress_score(score)
        print(f"  Score {score:.2f} → {result['emoji']} {result['level']}: {result['message']}")