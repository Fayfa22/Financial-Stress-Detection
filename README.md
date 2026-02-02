# Financial-Stress-Detection
## 📌 Project Overview
This project aims to estimate the financial stress level of companies by combining:
- quantitative financial indicators (accounting ratios)
- qualitative financial language analysis

The approach relies exclusively on classical machine learning and statistical methods, without using deep learning models.

---

## 🎯 Objectives
- Build a numerical financial stress score using accounting data
- Build a textual stress score using financial language
- Combine both scores into a unified stress index
- Analyze consistency and divergence between textual and numerical stress signals

---

## 📊 Datasets

### 1. Financial Data (Numerical)
- Polish Companies Bankruptcy Dataset (UCI)
- Source: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy
- Features: Financial ratios
- Target: Bankruptcy (used only for evaluation)

### 2. Financial Text Data
- Financial PhraseBank (Hugging Face)
- Source: https://huggingface.co/datasets/financial_phrasebank
- Features: Financial sentences labeled by sentiment

---

## 🛠 Technologies Used
- Python 3
- pandas, numpy
- scikit-learn
- scipy, statsmodels
- matplotlib, seaborn

---

## 🧠 Methodology

### Step 1: Data Preprocessing
- Missing value handling
- Feature normalization
- Text cleaning (lowercase, stopwords, lemmatization)

### Step 2: Numerical Stress Modeling
- Feature selection
- PCA for dimensionality reduction
- Logistic Regression / Random Forest
- Output: Financial stress probability score

### Step 3: Textual Stress Modeling
- TF-IDF vectorization
- Sentiment classification using Logistic Regression or SVM
- Output: Textual stress score

### Step 4: Stress Fusion
- Weighted aggregation of numerical and textual scores
- Divergence analysis (text vs numbers)

---

## 📈 Evaluation
- Correlation analysis between stress scores
- Comparison with bankruptcy labels
- Visualization of stress distributions

---

## 📌 Conclusion
This project demonstrates that meaningful financial stress indicators can be built using classical machine learning methods, combining quantitative and qualitative financial signals.

## Team
Hachicha Fayfa
Berred Kenza
