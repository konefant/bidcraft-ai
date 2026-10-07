"""
Routes API de BidCraft AI.

Rôle : exposer le pipeline RAG via HTTP. Ce module ne contient que la
logique "web" (validation, codes d'erreur) ; le travail de fond est délégué
aux modules app.rag.*
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.config import settings
from app.rag.pdf_parser import PDFParsingError, extract_text_from_pdf
from app.rag.rag_chain import (
    generate_technical_memo_section,
    index_chunks,
    retrieve_context,
    split_text,
)

router = APIRouter(prefix="/api", tags=["memo"])

# Consigne par défaut envoyée au LLM (à rendre configurable côté frontend plus tard)
DEFAULT_MEMO_PROMPT: str = (
    "À partir du cahier des charges (CCTP), rédige une section de mémoire "
    "technique présentant la compréhension du besoin, la méthodologie "
    "d'exécution proposée et les moyens mis en œuvre pour répondre aux "
    "exigences du marché."
)


class GenerateMemoResponse(BaseModel):
    """Schéma de la réponse JSON renvoyée au frontend."""

    filename: str
    chunks_count: int
    memo: str


@router.post("/generate-memo", response_model=GenerateMemoResponse)
async def generate_memo(file: UploadFile = File(...)) -> GenerateMemoResponse:
    """
    Reçoit un PDF (DCE / CCTP) et renvoie un mémoire technique généré.

    Pipeline : lecture -> extraction texte -> chunking -> indexation
    -> récupération du contexte -> génération LLM.
    """
    # 1. Validation du fichier
    filename = file.filename or "document.pdf"
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Seuls les fichiers PDF sont acceptés.",
        )

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux (max {settings.MAX_UPLOAD_SIZE_MB} Mo).",
        )

    # 2. Extraction du texte (CPU-bound -> exécutée hors de la boucle async)
    try:
        text = await run_in_threadpool(extract_text_from_pdf, content)
    except PDFParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    # 3. Découpage + indexation vectorielle (appels réseau bloquants)
    chunks = split_text(text)
    store = await run_in_threadpool(index_chunks, chunks, filename)

    # 4. Récupération du contexte pertinent
    context = await run_in_threadpool(
        retrieve_context, DEFAULT_MEMO_PROMPT, chunks, store
    )

    # 5. Génération du paragraphe de mémoire technique
    try:
        memo = await generate_technical_memo_section(DEFAULT_MEMO_PROMPT, context)
    except Exception as exc:  # erreurs OpenAI : clé invalide, quota, réseau...
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erreur lors de la génération par le LLM : {exc}",
        ) from exc

    return GenerateMemoResponse(filename=filename, chunks_count=len(chunks), memo=memo) 