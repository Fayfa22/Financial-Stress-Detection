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
from pathlib import Path

import os
# Ajouter src au path (support local et Docker)
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append('/app/src')

from schemas import (
    NumericalInput, TextInput, FusedInput,
    NumericalPrediction, TextPrediction, FusedPrediction,
    HealthResponse
)
from predict import (
    load_models, check_models_loaded,
    predict_numerical, predict_text, predict_fused
)

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
    
    - **`POST /predict/numerical`** : Prédiction à partir de 64 ratios financiers
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
    """Charger les modèles au démarrage de l'API"""
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
    """Page d'accueil de l'API"""
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
    """Vérifier l'état de l'API et le chargement des modèles"""
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


@app.post("/predict/numerical",
          response_model=NumericalPrediction,
          tags=["Prédictions"])
async def predict_num_endpoint(
    file: UploadFile = File(...),
):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        # ── JSON ──────────────────────────────────────────
        if filename.endswith('.json'):
            data = json.loads(contents.decode('utf-8'))
            if "features" not in data:
                raise ValueError("JSON doit contenir une clé 'features'")
            features = [float(x) for x in data["features"]]

        # ── CSV ───────────────────────────────────────────
        elif filename.endswith('.csv'):
            import io
            df = pd.read_csv(io.BytesIO(contents))
            features = df.iloc[0].tolist()

        # ── EXCEL ─────────────────────────────────────────
        elif filename.endswith(('.xlsx', '.xls')):
            import io
            df = pd.read_excel(io.BytesIO(contents))
            features = df.iloc[0].tolist()

        # ── ARFF ──────────────────────────────────────────
        elif filename.endswith('.arff'):
            import io
            from scipy.io import arff
            data, meta = arff.loadarff(io.StringIO(contents.decode('utf-8')))
            df = pd.DataFrame(data)
            # Exclure la colonne cible si présente
            numeric_cols = df.select_dtypes(include='number').columns
            features = df[numeric_cols].iloc[0].tolist()

        else:
            raise HTTPException(
                status_code=400,
                detail="Format non supporté. Utilisez : .json, .csv, .xlsx ou .arff"
            )

        # ── Validation 64 features ────────────────────────
        features = [float(x) for x in features]
        if len(features) != 64:
            raise HTTPException(
                status_code=400,
                detail=f"Attendu 64 features, reçu {len(features)}"
            )

        return predict_numerical(features)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

@app.post("/predict/text",
          response_model=TextPrediction,
          tags=["Prédictions"],
          summary="Prédiction textuelle (sentiment financier)")
def predict_text_endpoint(input_data: TextInput):
    """
    ## 📝 Prédire le stress financier à partir d'un texte financier
    
    **Entrée** : Texte financier (rapport, news, communiqué, etc.)
    
    **Sortie** : Sentiment, score de stress textuel, probabilités par classe
    
    ### Exemples de textes
    - "The company reported strong earnings with revenue growth."
    - "Declining profits and increasing debt raise concerns."
    - "Restructuring efforts show promising results."
    """
    try:
        result = predict_text(input_data.text)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fused",
          response_model=FusedPrediction,
          tags=["Prédictions"],
          summary="Prédiction fusionnée (fichier JSON + texte)")
async def predict_fused_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...),
    weight_num: float = Form(0.6),
    weight_text: float = Form(0.4),
):
    """
    Accepte :
    - Un fichier JSON avec {"features": [64 floats]}
    - Le texte en champ Form
    - Les poids en champs Form
    """
    try:
        # Validation poids
        total = weight_num + weight_text
        if abs(total - 1.0) > 0.01:
            weight_num = weight_num / total
            weight_text = weight_text / total

        # Lecture fichier
        contents = await file.read()
        data = json.loads(contents.decode('utf-8'))

        if "features" not in data or not isinstance(data["features"], list):
            raise ValueError("JSON invalide : clé 'features' manquante ou pas une liste")

        features = [float(x) for x in data["features"]]
        if len(features) != 64:
            raise HTTPException(status_code=400, detail=f"Attendu 64 features, reçu {len(features)}")

        result = predict_fused(
            features=features,
            text=text,
            weight_num=weight_num,
            weight_text=weight_text
        )
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Fichier JSON invalide")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
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