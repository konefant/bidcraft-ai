"""
Pipeline RAG (Retrieval-Augmented Generation) de BidCraft AI.

Étapes :
  1. split_text()                      -> découpe le texte en morceaux (chunks)
  2. index_chunks()                    -> embeddings + insertion dans pgvector
                                          (ou simulation si DATABASE_URL absent)
  3. retrieve_context()                -> récupère les passages pertinents
  4. generate_technical_memo_section() -> le LLM rédige le paragraphe final
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Limite de caractères de contexte envoyés au LLM (évite de dépasser la fenêtre)
MAX_CONTEXT_CHARS: int = 12_000


# --------------------------------------------------------------------------- #
# 1. Découpage                                                                #
# --------------------------------------------------------------------------- #
def split_text(text: str) -> list[str]:
    """
    Découpe un long texte en chunks qui se chevauchent légèrement.

    RecursiveCharacterTextSplitter essaie de couper d'abord sur les
    paragraphes, puis les lignes, puis les phrases : les chunks gardent
    ainsi un sens cohérent, ce qui améliore la qualité de la recherche.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# --------------------------------------------------------------------------- #
# 2. Embeddings + base vectorielle                                            #
# --------------------------------------------------------------------------- #
def get_embeddings() -> OpenAIEmbeddings:
    """Modèle d'embeddings : convertit un texte en vecteur numérique."""
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
    )


def _normalize_pg_url(url: str) -> str:
    """langchain-postgres exige le driver psycopg3 : postgresql+psycopg://"""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_vector_store(collection_name: Optional[str] = None) -> Optional[VectorStore]:
    """
    Retourne le vector store pgvector (Supabase), ou None en mode simulation.

    Conseil multi-tenant : utilisez une collection par entreprise cliente
    (ex. f"company_{company_id}") pour isoler les bases de connaissances.
    """
    if not settings.DATABASE_URL:
        return None

    # Import tardif : le package n'est requis que si la base est configurée.
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name or settings.VECTOR_COLLECTION_NAME,
        connection=_normalize_pg_url(settings.DATABASE_URL),
        use_jsonb=True,
    )


def index_chunks(
    chunks: list[str],
    source_name: str,
    collection_name: Optional[str] = None,
) -> Optional[VectorStore]:
    """
    Génère les embeddings des chunks et les insère dans pgvector.

    Mode simulation : si DATABASE_URL n'est pas défini, rien n'est envoyé à
    OpenAI ni à la base ; on journalise seulement l'opération. Pratique pour
    développer sans infrastructure.

    Returns:
        Le vector store utilisé, ou None en mode simulation.
    """
    store = get_vector_store(collection_name)

    if store is None:
        logger.info(
            "[SIMULATION] %d chunks de '%s' non insérés (DATABASE_URL absent).",
            len(chunks),
            source_name,
        )
        return None

    documents = [
        Document(page_content=chunk, metadata={"source": source_name, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    store.add_documents(documents)
    logger.info("%d chunks de '%s' insérés dans pgvector.", len(documents), source_name)
    return store


# --------------------------------------------------------------------------- #
# 3. Récupération du contexte                                                 #
# --------------------------------------------------------------------------- #
def retrieve_context(
    query: str,
    chunks: list[str],
    store: Optional[VectorStore] = None,
    k: int = 6,
) -> str:
    """
    Construit le contexte à fournir au LLM.

    - Avec un vector store : recherche par similarité sémantique (top-k).
    - Sans vector store (simulation) : on prend simplement les k premiers chunks.
    """
    if store is not None:
        docs = store.similarity_search(query, k=k)
        passages = [doc.page_content for doc in docs]
    else:
        passages = chunks[:k]

    return "\n\n---\n\n".join(passages)[:MAX_CONTEXT_CHARS]


# --------------------------------------------------------------------------- #
# 4. Génération                                                               #
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT: str = (
    "Tu es un rédacteur expert de mémoires techniques pour appels d'offres "
    "dans le BTP. Tu rédiges en français, dans un style professionnel, précis "
    "et structuré, adapté à un maître d'ouvrage public ou privé.\n"
    "Règles strictes :\n"
    "- Appuie-toi UNIQUEMENT sur le contexte fourni.\n"
    "- N'invente jamais de certification, de chiffre, de référence chantier "
    "ou de matériel absent du contexte.\n"
    "- Si une information manque, signale-le entre crochets, ex. "
    "[À compléter : certification Qualibat].\n"
    "- Réponds par un paragraphe rédigé, sans titre ni liste à puces."
)

_USER_PROMPT: str = (
    "Contexte (extraits du cahier des charges et de la base de connaissances) :\n"
    "{context}\n\n"
    "Consigne de rédaction :\n{prompt}"
)


async def generate_technical_memo_section(prompt: str, context: str) -> str:
    """
    Génère une section de mémoire technique avec le LLM.

    Args:
        prompt: la consigne (ex. « Rédige la méthodologie d'exécution du gros œuvre »).
        context: les passages pertinents (CCTP + base de connaissances).

    Returns:
        Le paragraphe rédigé, prêt à être inséré dans le mémoire.
    """
    llm = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
    )
    template = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _USER_PROMPT)]
    )
    # Chaîne LCEL : prompt -> LLM -> conversion de la réponse en str
    chain = template | llm | StrOutputParser()

    return (await chain.ainvoke({"prompt": prompt, "context": context})).strip()