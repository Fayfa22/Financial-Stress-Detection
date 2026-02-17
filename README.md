# 💰 Financial Stress Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.9.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)

<br/>

> **Estimating the financial stress level of companies by combining quantitative financial indicators and qualitative financial language analysis — using classical machine learning only.**

<br/>

[📊 Overview](#-project-overview) • [🎯 Objectives](#-objectives) • [📁 Structure](#-project-structure) • [🚀 Installation](#-installation) • [🧠 Methodology](#-methodology) • [📈 Results](#-results) • [👩‍💻 Author](#-author)

</div>

---

## 📌 Project Overview

**Financial Stress Detection** is a machine learning project that builds a unified financial stress index by combining two complementary signals:

- 📊 **Quantitative signal** — Financial ratios from accounting data (Polish bankruptcy dataset)
- 📝 **Qualitative signal** — Sentiment analysis of financial text (Financial PhraseBank)

The fusion of both signals produces a robust stress score that captures what numbers alone or text alone cannot.

> ⚠️ This project uses **exclusively classical ML methods** — no deep learning.

---

## 🎯 Objectives

| #   | Objective                                                     | Status         |
| --- | ------------------------------------------------------------- | -------------- |
| 1   | Build a numerical financial stress score from accounting data | ✅ Done        |
| 2   | Build a textual stress score from financial language          | ✅ Done        |
| 3   | Combine both scores into a unified stress index               | ✅ Done        |
| 4   | Analyze consistency & divergence between signals              | ✅ Done        |
| 5   | Expose predictions via a REST API (FastAPI)                   | 🔄 In progress |

---

## 📊 Datasets

### 1. 🏭 Financial Data (Numerical)

| Property          | Details                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| **Name**          | Polish Companies Bankruptcy Dataset                                                                    |
| **Source**        | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy) |
| **Files**         | `1year.arff` → `5year.arff`                                                                            |
| **Features**      | 64 financial ratios per company                                                                        |
| **Target**        | Bankruptcy label (0 = healthy, 1 = bankrupt)                                                           |
| **Total samples** | ~43,000 companies across 5 years                                                                       |

### 2. 📰 Financial Text Data

| Property     | Details                                                              |
| ------------ | -------------------------------------------------------------------- |
| **Name**     | Financial PhraseBank                                                 |
| **Source**   | [Hugging Face](https://huggingface.co/datasets/financial_phrasebank) |
| **Files**    | `train.parquet`, `test.parquet`                                      |
| **Features** | Financial sentences with sentiment labels                            |
| **Classes**  | Negative (0), Neutral (1), Positive (2)                              |

---

## 📁 Project Structure

```
financial_stress_project/
│
├── 📂 data/
│   ├── num_data/                  # ARFF files (1year → 5year)
│   └── text_data/                 # Parquet files (train/test)
│
├── 📂 src/
│   ├── config.py                  # Global configuration & paths
│   ├── load_data.py               # Data loading (ARFF + Parquet)
│   ├── preprocess_num.py          # Numerical preprocessing
│   ├── preprocess_text.py         # Text cleaning & lemmatization
│   ├── eda_num.py                 # Numerical exploratory analysis
│   ├── eda_text.py                # Textual exploratory analysis
│   ├── vectorize_text.py          # TF-IDF vectorization
│   ├── train_num_model.py         # RF + LR training with MLflow
│   ├── train_text_model.py        # LR + SVM training with MLflow
│   └── fusion_score.py            # Score fusion & divergence analysis
│
├── 📂 api/                        # FastAPI (coming soon)
│   ├── main_api.py
│   ├── schemas.py
│   └── predict.py
│
├── 📂 notebooks/
│   └── EDA_and_Vectorization.ipynb
│
├── 📂 outputs/
│   ├── figures/                   # Generated plots
│   ├── processed_data/            # Cleaned CSVs
│   └── vectorized_data/           # TF-IDF matrices (.npz)
│
├── 📂 models/                     # Trained models (.pkl) — not tracked in Git
├── 📂 mlruns/                     # MLflow experiments — not tracked in Git
│
├── main.py                        # Full pipeline entry point
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Fayfa22/Financial-Stress-Detection.git
cd Financial-Stress-Detection
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux / Mac
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources

```bash
python -m nltk.downloader punkt stopwords wordnet
```

### 5. Run the full pipeline

```bash
python main.py
```

---

## 🧠 Methodology

```
Raw Data
   │
   ├──► Numerical (ARFF)          ├──► Text (Parquet)
   │         │                    │         │
   │    Preprocessing             │    Cleaning & Lemmatization
   │    (imputation, scaling)     │    (NLTK pipeline)
   │         │                    │         │
   │    EDA & Visualization       │    EDA & Word Clouds
   │         │                    │         │
   │    SMOTE (rebalancing)       │    TF-IDF Vectorization
   │         │                    │         │
   │    Random Forest             │    Logistic Regression
   │    Logistic Regression       │    SVM (LinearSVC)
   │         │                    │         │
   │    Numerical Stress Score    │    Textual Stress Score
   │         │                    │         │
   └─────────┴────────────────────┘
                     │
              Weighted Fusion
           (60% numerical + 40% textual)
                     │
           Unified Stress Index [0, 1]
                     │
           ┌─────────┴──────────┐
         REST API            Divergence
        (FastAPI)             Analysis
```

### Step 1 — Preprocessing

- **Numerical** : Missing value imputation (median), StandardScaler normalization, SMOTE for class imbalance
- **Textual** : Lowercasing, URL/number removal, stopword filtering, WordNet lemmatization

### Step 2 — Numerical Stress Modeling

| Model               | Technique                          | Tuning                   |
| ------------------- | ---------------------------------- | ------------------------ |
| Random Forest       | 100–200 trees, max_depth 5–None    | GridSearchCV (5-fold CV) |
| Logistic Regression | L2 penalty, C ∈ {0.01, 0.1, 1, 10} | GridSearchCV (5-fold CV) |

### Step 3 — Textual Stress Modeling

| Model               | Technique                         | Tuning                   |
| ------------------- | --------------------------------- | ------------------------ |
| Logistic Regression | TF-IDF (5000 features, 1-2 grams) | GridSearchCV (5-fold CV) |
| SVM (LinearSVC)     | Calibrated for probabilities      | Fixed C=1                |

### Step 4 — Score Fusion

```
Stress_Index = 0.6 × Numerical_Score + 0.4 × Textual_Score
```

Divergence analysis flags when `|Numerical - Textual| > 0.4`.

### Stress Level Interpretation

| Score Range | Level       | Meaning                                      |
| ----------- | ----------- | -------------------------------------------- |
| 0.0 – 0.2   | 🟢 FAIBLE   | Healthy financial situation                  |
| 0.2 – 0.4   | 🟡 MODÉRÉ   | Minor tensions, monitoring advised           |
| 0.4 – 0.6   | 🟠 ÉLEVÉ    | Significant stress, corrective action needed |
| 0.6 – 0.8   | 🔴 CRITIQUE | Critical stress, urgent intervention         |
| 0.8 – 1.0   | ⛔ EXTRÊME  | Imminent bankruptcy risk                     |

---

## 📈 Results

All experiments are tracked with **MLflow**. To launch the UI:

```bash
mlflow ui
# Open: http://127.0.0.1:5000
```

### Outputs generated

| Folder                     | Content                                                                  |
| -------------------------- | ------------------------------------------------------------------------ |
| `outputs/figures/`         | Confusion matrices, feature importance, word clouds, score distributions |
| `outputs/processed_data/`  | Cleaned CSVs, descriptive statistics                                     |
| `outputs/vectorized_data/` | TF-IDF sparse matrices (`.npz`), fitted vectorizer (`.pkl`)              |

---

## 🛠️ Technologies

| Category          | Tools                            |
| ----------------- | -------------------------------- |
| **Language**      | Python 3.11                      |
| **Data**          | pandas, numpy, scipy, pyarrow    |
| **ML**            | scikit-learn, imbalanced-learn   |
| **NLP**           | nltk, TF-IDF                     |
| **Tracking**      | MLflow 3.9.0                     |
| **API**           | FastAPI, uvicorn _(coming soon)_ |
| **Visualization** | matplotlib, seaborn, wordcloud   |

---

## 🔄 Roadmap

- [x] EDA (numerical + textual)
- [x] Preprocessing pipeline
- [x] TF-IDF vectorization
- [x] Model training with MLflow tracking
- [x] Score fusion & divergence analysis
- [ ] FastAPI REST endpoints
- [ ] Swagger UI tests
- [ ] Docker containerization

---

## 👩‍💻 Author

<div align="center">

|     | Name               | University                   |
| --- | ------------------ | ---------------------------- |
| 👩‍💻  | **Hachicha Fayfa** | Tek-Up University — 2nd Year |

[![GitHub](https://img.shields.io/badge/GitHub-Fayfa22-181717?style=for-the-badge&logo=github)](https://github.com/Fayfa22)

</div>

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For any questions, please open an issue on [GitHub](https://github.com/Fayfa22/Financial-Stress-Detection/issues).

---

<div align="center">
<sub>Built with ❤️ at Tek-Up University</sub>
</div>
