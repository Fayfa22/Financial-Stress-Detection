# 💰 FinStress AI — Financial Stress Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.9.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)

<br/>

> **Academic Project — 2nd Year Engineering Cycle in Data Science & AI, Tek-Up University**
>
> A multimodal AI platform that detects financial stress in companies by fusing quantitative financial ratios and qualitative NLP signals — powered by classical machine learning, explainable AI (SHAP + TF-IDF), and a full-stack Docker deployment.

<br/>

[📌 Overview](#-project-overview) • [🎯 Objectives](#-objectives) • [🖥️ Interface](#️-interface) • [📁 Structure](#-project-structure) • [🚀 Quick Start](#-quick-start) • [🧠 Methodology](#-methodology) • [📈 Results](#-results) • [🔍 XAI](#-explainable-ai) • [👩‍💻 Author](#-author)

</div>

---

## 📌 Project Overview

**FinStress AI** is an end-to-end machine learning platform that builds a unified financial stress index by combining two complementary signals:

- 📊 **Quantitative signal** — 64 financial ratios from accounting data (Polish Bankruptcy Dataset)
- 📝 **Qualitative signal** — Sentiment analysis of financial text (Financial PhraseBank)

The fusion produces a robust stress score **[0, 1]** that captures what numbers alone or text alone cannot. The platform includes a full React frontend, a FastAPI backend, explainability via SHAP and TF-IDF, and is fully containerized with Docker.

> ⚠️ This project uses **exclusively classical ML methods** — no deep learning, no LLMs.

---

## 🎯 Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build a numerical financial stress score from accounting data | ✅ Done |
| 2 | Build a textual stress score from financial language | ✅ Done |
| 3 | Combine both scores into a unified stress index | ✅ Done |
| 4 | Analyze consistency & divergence between signals | ✅ Done |
| 5 | Expose predictions via a REST API (FastAPI) | ✅ Done |
| 6 | Build a React frontend with interactive UI | ✅ Done |
| 7 | Add Explainable AI (SHAP + TF-IDF visualization) | ✅ Done |
| 8 | Containerize with Docker Compose | ✅ Done |

---

## 🖥️ Interface

The platform offers three analysis modes accessible at `http://localhost:3000` :

| Mode | Description | Input |
|------|-------------|-------|
| 📝 **Textual Analysis** | NLP-based stress detection from financial text | Free text (annual report, press release) |
| 📊 **Numerical Analysis** | Quantitative stress score from financial ratios | File upload (.json, .csv, .xlsx, .arff) |
| 🔀 **Multimodal Fusion** | Combined score with adjustable weighting | Text + File |

After each prediction, the **🔍 Explain Decision (XAI)** button reveals:
- **SHAP waterfall chart** — which financial ratios drove the score
- **TF-IDF keyword chips** — which words most influenced the NLP prediction
- **Divergence alert** — when numerical and textual signals strongly disagree

### Stress Level Interpretation

| Score | Level | Meaning |
|-------|-------|---------|
| 0.0 – 0.2 | 🟢 FAIBLE | Healthy financial situation |
| 0.2 – 0.4 | 🟡 MODÉRÉ | Minor tensions, monitoring advised |
| 0.4 – 0.6 | 🟠 ÉLEVÉ | Significant stress, corrective action needed |
| 0.6 – 0.8 | 🔴 CRITIQUE | Critical stress, urgent intervention |
| 0.8 – 1.0 | ⛔ EXTRÊME | Imminent bankruptcy risk |

---

## 📊 Datasets

### 1. 🏭 Financial Data (Numerical)

| Property | Details |
|----------|---------|
| **Name** | Polish Companies Bankruptcy Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy) |
| **Files** | `1year.arff` → `5year.arff` |
| **Features** | 64 financial ratios per company |
| **Target** | Bankruptcy label (0 = healthy, 1 = bankrupt) |
| **Total samples** | ~43,000 companies across 5 years |

### 2. 📰 Financial Text Data

| Property | Details |
|----------|---------|
| **Name** | Financial PhraseBank |
| **Source** | [Hugging Face](https://huggingface.co/datasets/financial_phrasebank) |
| **Files** | `train.parquet`, `test.parquet` |
| **Features** | Financial sentences with sentiment labels |
| **Classes** | Negative (0), Neutral (1), Positive (2) |

---

## 📁 Project Structure

```
financial_stress_project/
│
├── 📂 api/
│   ├── main_api.py              # FastAPI app — all endpoints
│   ├── predict.py               # Prediction logic (num/text/fused)
│   ├── explain.py               # XAI — SHAP + TF-IDF explainability
│   ├── schemas.py               # Pydantic models
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile
│
├── 📂 frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Predict.tsx      # Main analysis page (3 modes + XAI)
│   │   │   └── Home.tsx         # Landing page
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   │   └── logo.svg             # FinStress AI favicon
│   └── Dockerfile
│
├── 📂 src/
│   ├── config.py                # Global configuration & paths
│   ├── load_data.py             # Data loading (ARFF + Parquet)
│   ├── preprocess_num.py        # Numerical preprocessing
│   ├── preprocess_text.py       # Text cleaning & lemmatization
│   ├── eda_num.py               # Numerical EDA
│   ├── eda_text.py              # Textual EDA
│   ├── vectorize_text.py        # TF-IDF vectorization
│   ├── train_num_model.py       # Random Forest training + MLflow
│   ├── train_text_model.py      # SVM/LR training + MLflow
│   └── fusion_score.py          # Score fusion & divergence
│
├── 📂 data/
│   ├── num_data/                # ARFF files (1year → 5year)
│   └── text_data/               # Parquet files (train/test)
│
├── 📂 models/                   # Trained .pkl files (git-ignored)
├── 📂 mlruns/                   # MLflow experiments (git-ignored)
├── 📂 outputs/                  # Figures, processed data, vectors
│
├── docker-compose.yml
├── main.py                      # Full training pipeline entry point
└── README.md
```

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Fayfa22/Financial-Stress-Detection.git
cd Financial-Stress-Detection

# 2. Train models first (required before running the API)
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py

# 3. Launch the full stack
docker-compose up --build

# 4. Open the app
# Frontend : http://localhost:3000
# API docs  : http://localhost:8000/docs
```

### Option B — Local Development

```bash
# Terminal 1 — API
cd api
pip install -r requirements.txt
uvicorn main_api:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Open: http://localhost:5173

# Terminal 3 — MLflow UI (optional)
mlflow ui
# Open: http://127.0.0.1:5000
```

---

## 🧠 Methodology

```
Raw Data
   │
   ├──► Numerical (ARFF)              ├──► Text (Parquet)
   │         │                        │         │
   │    Preprocessing                 │    Cleaning & Lemmatization
   │    (imputation, scaling,         │    (NLTK: stopwords,
   │     SMOTE rebalancing)           │     lemmatization, punkt)
   │         │                        │         │
   │    Random Forest                 │    TF-IDF (5000 features)
   │    (GridSearchCV)                │    LinearSVC + Calibration
   │         │                        │         │
   │    Numerical Score [0,1]         │    Textual Score [0,1]
   │         │                        │         │
   └─────────┴────────────────────────┘
                       │
              Weighted Fusion
        (default: 60% num + 40% text)
                       │
          Unified Stress Index [0, 1]
                       │
          ┌────────────┴────────────┐
      REST API                  XAI Layer
     (FastAPI)           (SHAP + TF-IDF chips)
          │
      React UI
  (3 modes + divergence alert)
```

### Models

| Signal | Model | Technique |
|--------|-------|-----------|
| Numerical | **Random Forest** | 100–200 trees, GridSearchCV 5-fold |
| Numerical | Logistic Regression | L2, C ∈ {0.01, 0.1, 1, 10} |
| Textual | **LinearSVC** (calibrated) | TF-IDF 5000 features, 1-2 grams |
| Textual | Logistic Regression | Baseline comparison |

### Fusion Formula

```
Stress_Index = w_num × Numerical_Score + w_text × Textual_Score
```

Divergence alert triggers when `|Numerical_Score − Textual_Score| > 0.4`.

---

## 🔍 Explainable AI

FinStress AI integrates two XAI techniques to make predictions transparent:

### 📊 SHAP (Numerical)
Uses `TreeExplainer` on the Random Forest to compute the contribution of each of the 64 financial ratios to the final score. Results displayed as a horizontal bar chart — red bars increase stress risk, green bars decrease it.

### 📝 TF-IDF Feature Importance (Textual)
Extracts the top-10 most influential words from the vectorized text input. Displayed as color-coded chips — larger/darker chips = higher influence on the prediction.

---

## 📈 Results & Tracking

All experiments tracked with **MLflow**:

```bash
mlflow ui   # http://127.0.0.1:5000
```

| Output | Location |
|--------|----------|
| Confusion matrices, feature importance | `outputs/figures/` |
| Cleaned datasets | `outputs/processed_data/` |
| TF-IDF sparse matrices | `outputs/vectorized_data/` |
| Trained models | `models/*.pkl` |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.11, TypeScript |
| **ML / NLP** | scikit-learn 1.8.0, nltk, imbalanced-learn, SHAP |
| **API** | FastAPI 0.109, uvicorn, pydantic |
| **Frontend** | React 19, MUI, Chart.js, Vite |
| **Tracking** | MLflow 3.9.0 |
| **DevOps** | Docker, Docker Compose, Nginx |
| **Data** | pandas, numpy, scipy, pyarrow |

---

## 🔄 Roadmap

- [x] EDA (numerical + textual)
- [x] Preprocessing pipeline
- [x] TF-IDF vectorization
- [x] Model training with MLflow tracking
- [x] Score fusion & divergence analysis
- [x] FastAPI REST endpoints
- [x] Swagger UI (`/docs`)
- [x] Docker Compose deployment
- [x] React frontend (3 analysis modes)
- [x] Explainable AI (SHAP + TF-IDF)
- [ ] Historical timeline (stress evolution over 5 years)
- [ ] Auto PDF report generation
- [ ] Company comparison mode

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health check |
| `POST` | `/predict/numerical` | Predict from financial ratios file |
| `POST` | `/predict/text` | Predict from financial text |
| `POST` | `/predict/fused` | Multimodal prediction (file + text) |
| `POST` | `/explain/numerical` | SHAP values for numerical prediction |
| `POST` | `/explain/text` | TF-IDF top words for text prediction |
| `POST` | `/explain/fused` | Full XAI (SHAP + TF-IDF combined) |

Full interactive documentation at **`http://localhost:8000/docs`**

---

## 🎓 Academic Context

This project was developed as part of the **2nd Year Engineering Cycle in Data Science & Artificial Intelligence** at **Tek-Up University**. It integrates skills acquired across the curriculum:

| Domain | Applied Skills |
|--------|---------------|
| **Machine Learning** | Supervised learning, model selection, cross-validation, SMOTE |
| **NLP** | Text preprocessing, TF-IDF vectorization, sentiment classification |
| **Software Engineering** | REST API design, modular architecture, Docker containerization |
| **Explainable AI** | SHAP values, feature importance, model transparency |
| **DevOps** | Docker Compose, Nginx reverse proxy, multi-service deployment |
| **Experiment Tracking** | MLflow runs, metrics logging, model versioning |

---

## 👩‍💻 Author

<div align="center">

| | Name | University | Program |
|-|------|------------|---------|
| 👩‍💻 | **Hachicha Fayfa** | Tek-Up University | 2nd Year Engineering Cycle — Data Science & AI |

[![GitHub](https://img.shields.io/badge/GitHub-Fayfa22-181717?style=for-the-badge&logo=github)](https://github.com/Fayfa22)

</div>

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
<sub>Built with ❤️ at Tek-Up University</sub>
</div>
