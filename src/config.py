import os
from pathlib import Path

# === CHEMINS DE BASE ===
BASE_DIR = Path(r"C:\Users\Anas\Desktop\bureau\fayfa\tek-up_2eme\financial_stress_project")

# Dossiers principaux
DATA_DIR = BASE_DIR / "data"
NUM_DATA_DIR = DATA_DIR / "num_data"
TEXT_DATA_DIR = DATA_DIR / "text_data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed_data"
VECTORIZED_DIR = OUTPUT_DIR / "vectorized_data"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlruns"

# Créer les dossiers s'ils n'existent pas
for directory in [OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR, VECTORIZED_DIR, MODELS_DIR, MLFLOW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === FICHIERS NUMÉRIQUES ===
NUM_FILES = {
    "1year": NUM_DATA_DIR / "1year.arff",
    "2year": NUM_DATA_DIR / "2year.arff",
    "3year": NUM_DATA_DIR / "3year.arff",
    "4year": NUM_DATA_DIR / "4year.arff",
    "5year": NUM_DATA_DIR / "5year.arff"
}

# === FICHIERS TEXTUELS ===
TEXT_FILES = {
    "train": TEXT_DATA_DIR / "train-00000-of-00001.parquet",
    "test": TEXT_DATA_DIR / "test-00000-of-00001.parquet"
}

# === PARAMÈTRES DE PRÉTRAITEMENT ===
# Numérique
MISSING_THRESHOLD = 0.4  # Supprimer colonnes avec >40% de valeurs manquantes
OUTLIER_METHOD = "iqr"   # 'iqr' ou 'zscore'
NORMALIZATION_METHOD = "standard"  # 'standard', 'minmax', ou 'robust'

# Texte
MIN_WORD_LENGTH = 2
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE = (1, 2)  # Unigrammes et bigrammes

# === PARAMÈTRES DE VISUALISATION ===
FIGSIZE_DEFAULT = (12, 6)
FIGSIZE_LARGE = (15, 10)
DPI = 300
STYLE = "whitegrid"

# === RANDOM STATE ===
RANDOM_STATE = 42

# === PARAMÈTRES MODÈLES ===
TEST_SIZE            = 0.2
CV_FOLDS             = 5

# === PARAMÈTRES MLFLOW ===
MLFLOW_TRACKING_URI    = MLFLOW_DIR.as_uri()
MLFLOW_EXPERIMENT_NUM  = "financial_stress_numerical"
MLFLOW_EXPERIMENT_TEXT = "financial_stress_textual"

# === PARAMÈTRES FUSION ===
WEIGHT_NUMERICAL = 0.6
WEIGHT_TEXTUAL   = 0.4

# Créer le dossier models
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Configuration chargée depuis : {BASE_DIR}")