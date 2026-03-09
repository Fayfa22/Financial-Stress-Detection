import shap
import numpy as np


def explain_with_shap(model, X, feature_names):
    """
    Calcule les valeurs SHAP pour un modèle RandomForest.
    Retourne le top 10 des features avec leur contribution.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Pour RandomForest : shap_values est une liste [classe_0, classe_1]
    if isinstance(shap_values, list):
        vals = shap_values[1][0]  # classe 1 (stress), première ligne
    else:
        vals = shap_values[0]

    importance = dict(zip(feature_names, vals))
    top10 = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    return [{"feature": k, "impact": round(float(v), 4)} for k, v in top10]


def get_base_value(model):
    """
    Retourne la valeur de base SHAP (score moyen du modèle).
    """
    explainer = shap.TreeExplainer(model)
    base = explainer.expected_value

    if isinstance(base, (list, np.ndarray)):
        return float(base[1])
    return float(base)


def explain_tfidf(vectorizer, text_clean):
    """
    Retourne les mots les plus influents selon TF-IDF.
    """
    X = vectorizer.transform([text_clean])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]

    word_importance = [
        (w, float(s))
        for w, s in zip(feature_names, tfidf_scores)
        if s > 0
    ]
    word_importance.sort(key=lambda x: x[1], reverse=True)

    return [{"word": w, "score": round(s, 4)} for w, s in word_importance[:10]]