"""
Point d'entrée de l'API BidCraft AI.

Lancer en local :
    uvicorn main:app --reload --port 8000
Documentation interactive : http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core.config import settings

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)

app = FastAPI(
    title=settings.APP_NAME,
    description="API de génération de mémoires techniques d'appels d'offres BTP.",
    version="0.1.0",
)

# CORS : autorise le frontend Next.js (http://localhost:3000) à appeler l'API
# depuis le navigateur. Sans cela, le navigateur bloque les requêtes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes (ex. POST /api/generate-memo)
app.include_router(api_router)


@app.get("/", tags=["health"])
async def health_check() -> dict[str, str]:
    """Endpoint de santé : vérifie que l'API répond."""
    return {"status": "ok", "app": "BidCraft AI API"}