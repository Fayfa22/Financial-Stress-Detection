"""
Entraînement des modèles textuels
Logistic Regression + SVM avec MLflow tracking
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing   import LabelEncoder
from scipy.sparse            import load_npz

import matplotlib.pyplot as plt
import seaborn as sns

import config


# ─────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES VECTORISÉES
# ─────────────────────────────────────────────────────────

def load_vectorized_data() -> tuple:
    """Charger les matrices TF-IDF et le vectorizer"""
    print("\n📂 Chargement des données vectorisées...")

    X_train    = load_npz(config.VECTORIZED_DIR / "X_train_tfidf.npz")
    X_test     = load_npz(config.VECTORIZED_DIR / "X_test_tfidf.npz")
    vectorizer = joblib.load(config.VECTORIZED_DIR / "vectorizer_tfidf.pkl")

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    return X_train, X_test, vectorizer


def load_text_labels() -> tuple:
    """Charger et encoder les labels train/test"""
    print("\n📂 Chargement des labels textuels...")

    train_df = pd.read_csv(config.PROCESSED_DIR / "processed_text_train.csv")
    test_df  = pd.read_csv(config.PROCESSED_DIR / "processed_text_test.csv")

    # Trouver la colonne label
    label_col = next(
        (c for c in ['label', 'sentiment', 'class'] if c in train_df.columns),
        train_df.columns[0]
    )
    print(f"  Colonne label : {label_col}")

    le      = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col].astype(str))
    y_test  = le.transform(test_df[label_col].astype(str))

    print(f"  Classes       : {le.classes_}")
    print(f"  Distribution train :\n{pd.Series(y_train).value_counts().to_string()}")

    return y_train, y_test, le


# ─────────────────────────────────────────────────────────
# 2. VISUALISATIONS
# ─────────────────────────────────────────────────────────

def plot_confusion_matrix_text(y_true, y_pred,
                                class_names: list, model_name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()

    path = config.FIGURES_DIR / f'cm_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────────────────
# 3. ENTRAÎNEMENT LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────

def train_logistic_regression_text(X_train, y_train,
                                    X_test, y_test,
                                    class_names: list) -> tuple:
    print("\n" + "="*60)
    print("  📈 LOGISTIC REGRESSION (Textuel)")
    print("="*60)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_TEXT)

    with mlflow.start_run(run_name="LogisticRegression_Text"):

        param_grid = {
            'C':           [0.1, 1, 10],
            'solver':      ['lbfgs'],
            'max_iter':    [1000],
        }

        print("  🔍 GridSearchCV en cours...")
        gs = GridSearchCV(
            LogisticRegression(
                random_state=config.RANDOM_STATE,
                class_weight='balanced'
            ),
            param_grid,
            cv=config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)

        best   = gs.best_estimator_
        params = gs.best_params_
        print(f"  ✅ Meilleurs params : {params}")

        y_pred  = best.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average='weighted')
        prec    = precision_score(y_test, y_pred, average='weighted')
        rec     = recall_score(y_test, y_pred, average='weighted')
        cv      = cross_val_score(best, X_train, y_train,
                                  cv=config.CV_FOLDS, scoring='f1_weighted')

        print(f"\n  📊 MÉTRIQUES :")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1        : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     CV F1     : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("cv_mean",   cv.mean())
        mlflow.log_metric("cv_std",    cv.std())

        cm_path = plot_confusion_matrix_text(
            y_test, y_pred, class_names, "Logistic Regression Text"
        )
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(best, "lr_text_model")

        joblib.dump(best, config.MODELS_DIR / "logistic_regression_text.pkl")
        print(f"  💾 Modèle sauvegardé : models/logistic_regression_text.pkl")
        print(f"  🔗 Run ID            : {mlflow.active_run().info.run_id}")

    return best, {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}


# ─────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT SVM
# ─────────────────────────────────────────────────────────

def train_svm_text(X_train, y_train,
                   X_test, y_test,
                   class_names: list) -> tuple:
    print("\n" + "="*60)
    print("  🔷 SVM — LinearSVC (Textuel)")
    print("="*60)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_TEXT)

    with mlflow.start_run(run_name="SVM_Text"):

        # CalibratedClassifierCV → permet predict_proba sur LinearSVC
        best_params = {'C': 1}
        best = CalibratedClassifierCV(
            LinearSVC(
                C=1,
                class_weight='balanced',
                random_state=config.RANDOM_STATE,
                max_iter=2000
            ),
            cv=3
        )

        print("  🔍 Entraînement en cours...")
        best.fit(X_train, y_train)

        y_pred = best.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average='weighted')
        prec   = precision_score(y_test, y_pred, average='weighted')
        rec    = recall_score(y_test, y_pred, average='weighted')
        cv     = cross_val_score(best, X_train, y_train,
                                 cv=config.CV_FOLDS, scoring='f1_weighted')

        print(f"\n  📊 MÉTRIQUES :")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1        : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     CV F1     : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("cv_mean",   cv.mean())
        mlflow.log_metric("cv_std",    cv.std())

        cm_path = plot_confusion_matrix_text(
            y_test, y_pred, class_names, "SVM Text"
        )
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(best, "svm_text_model")

        joblib.dump(best, config.MODELS_DIR / "svm_text.pkl")
        print(f"  💾 Modèle sauvegardé : models/svm_text.pkl")
        print(f"  🔗 Run ID            : {mlflow.active_run().info.run_id}")

    return best, {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}


# ─────────────────────────────────────────────────────────
# 5. PIPELINE COMPLET TEXTUEL
# ─────────────────────────────────────────────────────────

def train_all_text_models() -> dict:
    """Pipeline complet : charge, entraîne, compare, sauvegarde"""

    print("\n" + "="*70)
    print(" "*10 + "🤖 ENTRAÎNEMENT DES MODÈLES TEXTUELS")
    print("="*70)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    X_train, X_test, vectorizer = load_vectorized_data()
    y_train, y_test, le         = load_text_labels()
    class_names                 = list(le.classes_)

    results = {}

    lr_model, lr_metrics = train_logistic_regression_text(
        X_train, y_train, X_test, y_test, class_names
    )
    results['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}

    svm_model, svm_metrics = train_svm_text(
        X_train, y_train, X_test, y_test, class_names
    )
    results['svm'] = {'model': svm_model, 'metrics': svm_metrics}

    # ── Tableau comparatif ────────────────────────────────
    print("\n" + "="*70)
    print("  📊 COMPARAISON DES MODÈLES TEXTUELS")
    print("="*70)
    print(f"\n  {'Modèle':<28} {'Accuracy':>10} {'F1':>10} {'Recall':>10}")
    print("  " + "-"*58)

    best_name, best_f1 = None, 0
    for name, res in results.items():
        m = res['metrics']
        print(f"  {name:<28} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['recall']:>10.4f}")
        if m['f1'] > best_f1:
            best_f1, best_name = m['f1'], name

    print(f"\n  🏆 Meilleur modèle : {best_name}  (F1 = {best_f1:.4f})")

    joblib.dump(results[best_name]['model'], config.MODELS_DIR / "best_text_model.pkl")
    joblib.dump(le,                          config.MODELS_DIR / "label_encoder_text.pkl")
    joblib.dump(vectorizer,                  config.MODELS_DIR / "tfidf_vectorizer.pkl")
    print(f"  💾 best_text_model.pkl sauvegardé dans models/")

    return results


if __name__ == "__main__":
    train_all_text_models()