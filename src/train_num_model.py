"""
Entraînement des modèles numériques
Random Forest + Logistic Regression avec MLflow tracking
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

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.preprocessing   import LabelEncoder
from imblearn.over_sampling  import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

import config

# ─────────────────────────────────────────────────────────
# 1. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────

def prepare_numerical_data(num_data: dict) -> tuple:
    """
    Concaténer tous les fichiers annuels en un seul DataFrame
    et préparer X, y
    """
    print("\n📊 Préparation des données numériques...")

    dfs = []
    for year, df in num_data.items():
        df = df.copy()
        df['year'] = year
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Encoder la cible (0 = non-bankrupt, 1 = bankrupt)
    le = LabelEncoder()
    df_all['class'] = le.fit_transform(df_all['class'].astype(str))

    print(f"  Shape total      : {df_all.shape}")
    print(f"  Classes          : {le.classes_}")
    print(f"  Distribution     :\n{df_all['class'].value_counts().to_string()}")

    feature_cols = [c for c in df_all.columns if c not in ['class', 'year']]
    X = df_all[feature_cols].fillna(df_all[feature_cols].median())
    y = df_all['class']

    return X, y, le, feature_cols


def apply_smote(X_train, y_train) -> tuple:
    """
    SMOTE pour rééquilibrer les classes (très déséquilibrées en finance)
    """
    print("\n⚖️  Application de SMOTE...")
    print(f"  Avant : {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Après : {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────
# 2. VISUALISATIONS
# ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    """Sauvegarder et afficher la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-Bankrupt', 'Bankrupt'],
        yticklabels=['Non-Bankrupt', 'Bankrupt']
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()

    path = config.FIGURES_DIR / f'cm_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    return path


def plot_feature_importance(model, feature_names: list,
                             model_name: str, top_n: int = 20) -> Path | None:
    """Feature importance (Random Forest uniquement)"""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n)

    plt.figure(figsize=(9, 7))
    importances.plot(kind='barh', color='#3498db')
    plt.title(f'Top {top_n} Features — {model_name}')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = config.FIGURES_DIR / f'fi_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────────────────
# 3. ENTRAÎNEMENT RANDOM FOREST
# ─────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test,
                         feature_names: list) -> tuple:
    print("\n" + "="*60)
    print("  🌲 RANDOM FOREST")
    print("="*60)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NUM)

    with mlflow.start_run(run_name="RandomForest_Numerical"):

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth':    [5, 10, None],
            'class_weight': ['balanced']
        }

        print("  🔍 GridSearchCV en cours...")
        gs = GridSearchCV(
            RandomForestClassifier(random_state=config.RANDOM_STATE),
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

        # Prédictions
        y_pred       = best.predict(X_test)
        y_proba      = best.predict_proba(X_test)[:, 1]

        # Métriques
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        auc  = roc_auc_score(y_test, y_proba)
        cv   = cross_val_score(best, X_train, y_train,
                               cv=config.CV_FOLDS, scoring='f1_weighted')

        print(f"\n  📊 MÉTRIQUES :")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1        : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     ROC-AUC   : {auc:.4f}")
        print(f"     CV F1     : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Bankrupt','Bankrupt'])}")

        # ── MLflow logging ──────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("cv_mean",   cv.mean())
        mlflow.log_metric("cv_std",    cv.std())

        cm_path = plot_confusion_matrix(y_test, y_pred, "Random Forest")
        mlflow.log_artifact(str(cm_path))

        fi_path = plot_feature_importance(best, feature_names, "Random Forest")
        if fi_path:
            mlflow.log_artifact(str(fi_path))

        mlflow.sklearn.log_model(best, "random_forest_model")

        # ── Sauvegarde locale ────────────────────────────────
        joblib.dump(best, config.MODELS_DIR / "random_forest_num.pkl")
        print(f"  💾 Modèle sauvegardé : models/random_forest_num.pkl")
        print(f"  🔗 Run ID            : {mlflow.active_run().info.run_id}")

    return best, {'accuracy': acc, 'f1': f1,
                  'precision': prec, 'recall': rec, 'roc_auc': auc}


# ─────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────

def train_logistic_regression_num(X_train, y_train,
                                   X_test, y_test) -> tuple:
    print("\n" + "="*60)
    print("  📈 LOGISTIC REGRESSION (Numérique)")
    print("="*60)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NUM)

    with mlflow.start_run(run_name="LogisticRegression_Numerical"):

        param_grid = {
            'C':            [0.01, 0.1, 1, 10],
            'solver':       ['lbfgs'],
            'max_iter':     [1000],
            'class_weight': ['balanced']
        }

        print("  🔍 GridSearchCV en cours...")
        gs = GridSearchCV(
            LogisticRegression(random_state=config.RANDOM_STATE),
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
        y_proba = best.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        auc  = roc_auc_score(y_test, y_proba)
        cv   = cross_val_score(best, X_train, y_train,
                               cv=config.CV_FOLDS, scoring='f1_weighted')

        print(f"\n  📊 MÉTRIQUES :")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1        : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     ROC-AUC   : {auc:.4f}")
        print(f"     CV F1     : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Bankrupt','Bankrupt'])}")

        # ── MLflow logging ──────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("cv_mean",   cv.mean())
        mlflow.log_metric("cv_std",    cv.std())

        cm_path = plot_confusion_matrix(y_test, y_pred, "Logistic Regression Num")
        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(best, "logistic_regression_num_model")

        joblib.dump(best, config.MODELS_DIR / "logistic_regression_num.pkl")
        print(f"  💾 Modèle sauvegardé : models/logistic_regression_num.pkl")
        print(f"  🔗 Run ID            : {mlflow.active_run().info.run_id}")

    return best, {'accuracy': acc, 'f1': f1,
                  'precision': prec, 'recall': rec, 'roc_auc': auc}


# ─────────────────────────────────────────────────────────
# 5. PIPELINE COMPLET NUMÉRIQUE
# ─────────────────────────────────────────────────────────

def train_all_numerical_models(num_data: dict) -> dict:
    """Pipeline complet : prépare, entraîne, compare, sauvegarde"""

    print("\n" + "="*70)
    print(" "*10 + "🤖 ENTRAÎNEMENT DES MODÈLES NUMÉRIQUES")
    print("="*70)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    X, y, le, feature_names = prepare_numerical_data(num_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"\n  Train : {X_train.shape}  |  Test : {X_test.shape}")

    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    results = {}

    rf_model, rf_metrics = train_random_forest(
        X_train_bal, y_train_bal, X_test, y_test, feature_names
    )
    results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}

    lr_model, lr_metrics = train_logistic_regression_num(
        X_train_bal, y_train_bal, X_test, y_test
    )
    results['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}

    # ── Tableau comparatif ────────────────────────────────
    print("\n" + "="*70)
    print("  📊 COMPARAISON DES MODÈLES NUMÉRIQUES")
    print("="*70)
    print(f"\n  {'Modèle':<28} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("  " + "-"*58)

    best_name, best_f1 = None, 0
    for name, res in results.items():
        m = res['metrics']
        print(f"  {name:<28} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")
        if m['f1'] > best_f1:
            best_f1, best_name = m['f1'], name

    print(f"\n  🏆 Meilleur modèle : {best_name}  (F1 = {best_f1:.4f})")

    # Sauvegarder le meilleur
    joblib.dump(results[best_name]['model'], config.MODELS_DIR / "best_num_model.pkl")
    joblib.dump(le,                          config.MODELS_DIR / "label_encoder_num.pkl")
    print(f"  💾 best_num_model.pkl sauvegardé dans models/")

    return results


if __name__ == "__main__":
    from load_data      import load_all_numerical_data
    from preprocess_num import preprocess_numerical_data

    num_data      = load_all_numerical_data()
    processed     = preprocess_numerical_data(num_data)
    train_all_numerical_models(processed)