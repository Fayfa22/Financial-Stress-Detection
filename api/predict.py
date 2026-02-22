"""
Logique de prédiction pour l'API
Chargement des modèles et fonctions de prédiction
"""

import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import config
from preprocess_text import preprocess_text_pipeline
from fusion_score import fuse_scores, interpret_stress_score

# ══════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES (AU DÉMARRAGE)
# ══════════════════════════════════════════════════════════

MODELS = {}

def load_models():
    """Charger tous les modèles au démarrage de l'API"""
    global MODELS
    
    print("\n🔄 Chargement des modèles...")
    
    try:
        MODELS['num_model']  = joblib.load(config.MODELS_DIR / "best_num_model.pkl")
        MODELS['text_model'] = joblib.load(config.MODELS_DIR / "best_text_model.pkl")
        MODELS['vectorizer'] = joblib.load(config.MODELS_DIR / "tfidf_vectorizer.pkl")
        MODELS['le_text']    = joblib.load(config.MODELS_DIR / "label_encoder_text.pkl")
        MODELS['le_num']     = joblib.load(config.MODELS_DIR / "label_encoder_num.pkl")
        
        print("✅ Tous les modèles chargés avec succès")
        print(f"   - Modèle numérique : {type(MODELS['num_model']).__name__}")
        print(f"   - Modèle textuel   : {type(MODELS['text_model']).__name__}")
        
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles : {e}")
        return False


def check_models_loaded() -> bool:
    """Vérifier si les modèles sont chargés"""
    required = ['num_model', 'text_model', 'vectorizer', 'le_text', 'le_num']
    return all(key in MODELS for key in required)


# ══════════════════════════════════════════════════════════
# PRÉDICTION NUMÉRIQUE
# ══════════════════════════════════════════════════════════

def predict_numerical(features: list) -> dict:
    """
    Prédire le stress financier à partir de ratios financiers
    
    Args:
        features: Liste de 64 ratios financiers
    
    Returns:
        dict avec stress_score, prediction, confidence, interpretation
    """
    if not check_models_loaded():
        raise RuntimeError("Modèles non chargés. Redémarrez l'API.")
    
    model = MODELS['num_model']
    le    = MODELS['le_num']
    
    # Convertir en DataFrame
    X = pd.DataFrame([features])
    X = X.fillna(X.median())
    
    # Prédiction
    proba       = model.predict_proba(X)[0]
    pred_class  = model.predict(X)[0]
    stress_score = float(proba[1])  # Probabilité de faillite
    confidence   = float(max(proba))
    
    # Classe prédite
    prediction = str(le.inverse_transform([pred_class])[0])
    
    # Interprétation
    interpretation = interpret_stress_score(stress_score)
    
    return {
        'stress_score':   stress_score,
        'prediction':     prediction,
        'confidence':     confidence,
        'interpretation': interpretation
    }


# ══════════════════════════════════════════════════════════
# PRÉDICTION TEXTUELLE
# ══════════════════════════════════════════════════════════

def predict_text(text: str) -> dict:
    """
    Prédire le stress financier à partir d'un texte
    
    Args:
        text: Texte financier brut
    
    Returns:
        dict avec sentiment, stress_score, confidence, probabilities, interpretation
    """
    if not check_models_loaded():
        raise RuntimeError("Modèles non chargés. Redémarrez l'API.")
    
    model      = MODELS['text_model']
    vectorizer = MODELS['vectorizer']
    le         = MODELS['le_text']
    
    # Prétraiter le texte
    clean_text = preprocess_text_pipeline(text)
    
    if not clean_text or len(clean_text.strip()) < 3:
        raise ValueError("Le texte est vide après prétraitement. Essayez un texte plus long.")
    
    # Vectoriser
    X = vectorizer.transform([clean_text])
    
    # Prédiction
    proba     = model.predict_proba(X)[0]
    pred_idx  = model.predict(X)[0]
    sentiment = str(le.inverse_transform([pred_idx])[0])
    
    # Classes et probabilités
    classes      = list(le.classes_)
    probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
    
    # Trouver l'index de la classe négative
    neg_keywords = ['negative', 'neg', '0', 'bearish']
    neg_idx = 0
    for i, c in enumerate(classes):
        if any(k in str(c).lower() for k in neg_keywords):
            neg_idx = i
            break
    
    stress_score = float(proba[neg_idx])
    confidence   = float(max(proba))
    
    # Interprétation
    interpretation = interpret_stress_score(stress_score)
    
    return {
        'sentiment':      sentiment,
        'stress_score':   stress_score,
        'confidence':     confidence,
        'probabilities':  probabilities,
        'interpretation': interpretation
    }


# ══════════════════════════════════════════════════════════
# PRÉDICTION FUSIONNÉE
# ══════════════════════════════════════════════════════════

def predict_fused(features: list, text: str,
                  weight_num: float = 0.6,
                  weight_text: float = 0.4) -> dict:
    """
    Prédire le stress financier en combinant numérique + textuel
    
    Args:
        features: 64 ratios financiers
        text: Texte financier
        weight_num: Poids du score numérique
        weight_text: Poids du score textuel
    
    Returns:
        dict avec scores fusionnés, divergence, interprétation
    """
    # Prédictions individuelles
    num_result  = predict_numerical(features)
    text_result = predict_text(text)
    
    num_score  = num_result['stress_score']
    text_score = text_result['stress_score']
    
    # Fusion
    fused = float(fuse_scores(
        np.array([num_score]),
        np.array([text_score]),
        weight_num,
        weight_text
    )[0])
    
    # Divergence
    divergence = abs(num_score - text_score)
    
    # Alerte si divergence forte
    alert = None
    if divergence > 0.4:
        alert = (
            f"⚠️ Divergence forte ({divergence:.2f}) entre signal "
            f"numérique ({num_score:.2f}) et textuel ({text_score:.2f})"
        )
    
    # Interprétation
    interpretation = interpret_stress_score(fused)
    
    return {
        'numerical_score': num_score,
        'textual_score':   text_score,
        'fused_score':     fused,
        'weight_num':      weight_num,
        'weight_text':     weight_text,
        'divergence':      divergence,
        'interpretation':  interpretation,
        'alert':           alert
    }