"""
API FastAPI principale
Financial Stress Detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import sys
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
from explain import explain_with_shap, get_base_value, explain_tfidf

import os
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append('/app/src')

from schemas import (
    NumericalInput, TextInput, FusedInput,
    NumericalPrediction, TextPrediction, FusedPrediction,
    HealthResponse
)
from predict import (
    MODELS, load_models, check_models_loaded,
    predict_numerical, predict_text, predict_fused
)

# ══════════════════════════════════════════════════════════
# HELPER : parser n'importe quel fichier → 64 features
# ══════════════════════════════════════════════════════════

def parse_file_to_features(contents: bytes, filename: str) -> list:
    """Parse un fichier (json/csv/xlsx/arff) et retourne 64 features."""
    filename = filename.lower()

    if filename.endswith('.json'):
        data = json.loads(contents.decode('utf-8'))
        if "features" not in data:
            raise ValueError("JSON doit contenir une clé 'features'")
        features = [float(x) for x in data["features"]]

    elif filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
        numeric_cols = df.select_dtypes(include='number').columns
        features = df[numeric_cols].iloc[0].tolist()

    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(contents))
        numeric_cols = df.select_dtypes(include='number').columns
        features = df[numeric_cols].iloc[0].tolist()

    elif filename.endswith('.arff'):
        from scipy.io import arff
        data, meta = arff.loadarff(io.StringIO(contents.decode('utf-8')))
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include='number').columns
        features = df[numeric_cols].iloc[0].tolist()

    else:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Utilisez : .json, .csv, .xlsx ou .arff"
        )

    features = [float(x) for x in features]

    if len(features) < 64:
        features = features + [0.0] * (64 - len(features))
    elif len(features) > 64:
        features = features[:64]

    return features


# ══════════════════════════════════════════════════════════
# CRÉATION DE L'APPLICATION
# ══════════════════════════════════════════════════════════

app = FastAPI(
    title="💰 Financial Stress Detection API",
    description="""
    ## 📊 API de détection du stress financier
    
    Cette API combine **analyse numérique** (ratios financiers) et **analyse textuelle** 
    (sentiment financier) pour produire un score de stress unifié.
    
    ### 🎯 Endpoints disponibles
    
    - **`POST /predict/numerical`** : Prédiction à partir de ratios financiers
    - **`POST /predict/text`** : Prédiction à partir d'un texte financier
    - **`POST /predict/fused`** : Prédiction combinée (numérique + textuelle)
    - **`GET /health`** : Vérifier l'état de l'API
    
    ### 📈 Interprétation des scores
    
    | Score | Niveau | Signification |
    |-------|--------|---------------|
    | 0.0 - 0.2 | 🟢 FAIBLE | Situation saine |
    | 0.2 - 0.4 | 🟡 MODÉRÉ | Légères tensions |
    | 0.4 - 0.6 | 🟠 ÉLEVÉ | Stress significatif |
    | 0.6 - 0.8 | 🔴 CRITIQUE | Intervention urgente |
    | 0.8 - 1.0 | ⛔ EXTRÊME | Risque de faillite |
    
    ### 👩‍💻 Auteur
    **Hachicha Fayfa** - Tek-Up University
    
    [📂 GitHub Repository](https://github.com/Fayfa22/Financial-Stress-Detection)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Hachicha Fayfa",
        "url": "https://github.com/Fayfa22",
    },
    license_info={
        "name": "MIT License",
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print(" "*15 + "🚀 DÉMARRAGE DE L'API")
    print("="*60)
    success = load_models()
    if not success:
        print("\n⚠️  ATTENTION : Les modèles n'ont pas pu être chargés.")
        print("   Exécutez 'python main.py' pour entraîner les modèles d'abord.")
    print("\n" + "="*60)
    print(" "*10 + "✅ API prête sur http://127.0.0.1:8000")
    print(" "*10 + "📖 Documentation : http://127.0.0.1:8000/docs")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "💰 Financial Stress Detection API",
        "version": "1.0.0",
        "author": "Hachicha Fayfa",
        "university": "Tek-Up University",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_numerical": "/predict/numerical",
            "predict_text": "/predict/text",
            "predict_fused": "/predict/fused"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    models_ok = check_models_loaded()
    if models_ok:
        return {
            "status": "healthy",
            "models_loaded": True,
            "message": "API opérationnelle - Tous les modèles chargés"
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": False,
                "message": "Modèles non chargés. Exécutez 'python main.py' d'abord."
            }
        )


# ══════════════════════════════════════════════════════════
# PRÉDICTIONS
# ══════════════════════════════════════════════════════════

@app.post("/predict/numerical",
          response_model=NumericalPrediction,
          tags=["Prédictions"])
async def predict_num_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        features = parse_file_to_features(contents, file.filename)
        return predict_numerical(features)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/text",
          response_model=TextPrediction,
          tags=["Prédictions"])
async def predict_text_endpoint(input_data: TextInput):
    try:
        if not check_models_loaded():
            raise HTTPException(status_code=503, detail="Modèles non chargés")
        if not input_data.text or not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        return predict_text(input_data.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fused",
          response_model=FusedPrediction,
          tags=["Prédictions"],
          summary="Prédiction fusionnée (fichier + texte)")
async def predict_fused_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...),
    weight_num: float = Form(0.6),
    weight_text: float = Form(0.4),
):
    try:
        total = weight_num + weight_text
        if abs(total - 1.0) > 0.01:
            weight_num = weight_num / total
            weight_text = weight_text / total

        contents = await file.read()
        features = parse_file_to_features(contents, file.filename)

        result = predict_fused(
            features=features,
            text=text,
            weight_num=weight_num,
            weight_text=weight_text
        )
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
# EXPLICABILITÉ (XAI)
# ══════════════════════════════════════════════════════════

@app.post("/explain/numerical", tags=["Explicabilité"])
async def explain_numerical(file: UploadFile = File(...)):
    try:
        if not check_models_loaded():
            raise HTTPException(status_code=503, detail="Modèles non chargés")
        contents = await file.read()
        features = parse_file_to_features(contents, file.filename)
        feature_names = [f"ratio_{i+1}" for i in range(64)]
        X = pd.DataFrame([features])
        top10 = explain_with_shap(MODELS['num_model'], X.values, feature_names)
        base_value = get_base_value(MODELS['num_model'])
        return {"shap_values": top10, "base_value": base_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/text", tags=["Explicabilité"])
async def explain_text_endpoint(input_data: TextInput):
    try:
        if not check_models_loaded():
            raise HTTPException(status_code=503, detail="Modèles non chargés")
        from preprocess_text import preprocess_text_pipeline
        clean_text = preprocess_text_pipeline(input_data.text)
        top10 = explain_tfidf(MODELS['vectorizer'], clean_text)
        return {"top_words": top10, "cleaned_text": clean_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/fused", tags=["Explicabilité"])
async def explain_fused(file: UploadFile = File(...), text: str = Form(...)):
    try:
        if not check_models_loaded():
            raise HTTPException(status_code=503, detail="Modèles non chargés")
        # Numérique
        contents = await file.read()
        features = parse_file_to_features(contents, file.filename)
        feature_names = [f"ratio_{i+1}" for i in range(64)]
        X = pd.DataFrame([features])
        top10_shap = explain_with_shap(MODELS['num_model'], X.values, feature_names)
        base_value = get_base_value(MODELS['num_model'])
        # Textuel
        from preprocess_text import preprocess_text_pipeline
        clean_text = preprocess_text_pipeline(text)
        top10_words = explain_tfidf(MODELS['vectorizer'], clean_text)
        return {
            "numerical": {"shap_values": top10_shap, "base_value": base_value},
            "textual": {"top_words": top10_words, "cleaned_text": clean_text}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )