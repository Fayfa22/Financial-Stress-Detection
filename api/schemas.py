"""
Schémas Pydantic pour validation des requêtes/réponses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# ══════════════════════════════════════════════════════════
# SCHÉMAS D'ENTRÉE (REQUÊTES)
# ══════════════════════════════════════════════════════════

class NumericalInput(BaseModel):
    """Entrée pour prédiction numérique (ratios financiers)"""
    features: List[float] = Field(
        ...,
        description="Liste des 64 ratios financiers",
        min_length=64,
        max_length=64
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.8] + [0.1] * 61  # 64 valeurs
            }
        }


class TextInput(BaseModel):
    """Entrée pour prédiction textuelle (sentiment financier)"""
    text: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Texte financier à analyser"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The company reported significant losses this quarter with declining revenues and increased debt levels."
            }
        }


class FusedInput(BaseModel):
    """Entrée pour prédiction fusionnée (numérique + textuelle)"""
    features: List[float] = Field(
        ...,
        description="64 ratios financiers",
        min_length=64,
        max_length=64
    )
    text: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Texte financier associé"
    )
    weight_num: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Poids du score numérique (0-1)"
    )
    weight_text: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Poids du score textuel (0-1)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.8] + [0.1] * 61,
                "text": "Company facing severe financial difficulties with negative cash flow.",
                "weight_num": 0.6,
                "weight_text": 0.4
            }
        }


# ══════════════════════════════════════════════════════════
# SCHÉMAS DE SORTIE (RÉPONSES)
# ══════════════════════════════════════════════════════════

class InterpretationSchema(BaseModel):
    """Interprétation d'un score de stress"""
    level: str = Field(..., description="Niveau de stress (FAIBLE, MODÉRÉ, etc.)")
    emoji: str = Field(..., description="Emoji représentatif")
    message: str = Field(..., description="Message explicatif")


class NumericalPrediction(BaseModel):
    """Réponse pour prédiction numérique"""
    stress_score: float = Field(..., ge=0.0, le=1.0, description="Score de stress [0, 1]")
    prediction: str = Field(..., description="Classe prédite (Non-Bankrupt / Bankrupt)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance du modèle")
    interpretation: InterpretationSchema

    class Config:
        json_schema_extra = {
            "example": {
                "stress_score": 0.35,
                "prediction": "Non-Bankrupt",
                "confidence": 0.82,
                "interpretation": {
                    "level": "MODÉRÉ",
                    "emoji": "🟡",
                    "message": "Légères tensions. Surveillance recommandée."
                }
            }
        }


class TextPrediction(BaseModel):
    """Réponse pour prédiction textuelle"""
    sentiment: str = Field(..., description="Sentiment prédit (negative/neutral/positive)")
    stress_score: float = Field(..., ge=0.0, le=1.0, description="Score de stress textuel")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance du modèle")
    probabilities: Dict[str, float] = Field(..., description="Probabilités par classe")
    interpretation: InterpretationSchema

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "negative",
                "stress_score": 0.72,
                "confidence": 0.85,
                "probabilities": {
                    "negative": 0.72,
                    "neutral": 0.20,
                    "positive": 0.08
                },
                "interpretation": {
                    "level": "CRITIQUE",
                    "emoji": "🔴",
                    "message": "Stress critique. Intervention urgente."
                }
            }
        }


class FusedPrediction(BaseModel):
    """Réponse pour prédiction fusionnée"""
    numerical_score: float = Field(..., ge=0.0, le=1.0)
    textual_score: float = Field(..., ge=0.0, le=1.0)
    fused_score: float = Field(..., ge=0.0, le=1.0)
    weight_num: float = Field(..., ge=0.0, le=1.0)
    weight_text: float = Field(..., ge=0.0, le=1.0)
    divergence: float = Field(..., ge=0.0, description="Divergence entre scores")
    interpretation: InterpretationSchema
    alert: Optional[str] = Field(None, description="Alerte si divergence élevée")

    class Config:
        json_schema_extra = {
            "example": {
                "numerical_score": 0.25,
                "textual_score": 0.78,
                "fused_score": 0.46,
                "weight_num": 0.6,
                "weight_text": 0.4,
                "divergence": 0.53,
                "interpretation": {
                    "level": "ÉLEVÉ",
                    "emoji": "🟠",
                    "message": "Stress significatif. Action corrective conseillée."
                },
                "alert": "⚠️ Divergence forte (0.53) entre signal numérique (0.25) et textuel (0.78)"
            }
        }


class HealthResponse(BaseModel):
    """Réponse pour le health check"""
    status: str
    models_loaded: bool
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": True,
                "message": "API opérationnelle - Tous les modèles chargés"
            }
        }