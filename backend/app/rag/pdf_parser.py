"""
Extraction de texte depuis un PDF (DCE / CCTP) avec PyPDF.

Rôle : transformer les octets d'un PDF téléchargé en texte brut exploitable
par le pipeline RAG. Limite connue : les PDF scannés (images) ne contiennent
pas de texte ; il faudrait alors ajouter un OCR (ex. Tesseract) en amont.
"""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFParsingError(Exception):
    """Levée quand le PDF est illisible, protégé ou ne contient pas de texte."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extrait le texte brut d'un PDF.

    Args:
        file_bytes: contenu binaire du fichier PDF.

    Returns:
        Le texte de toutes les pages, séparées par un saut de ligne.

    Raises:
        PDFParsingError: PDF corrompu, chiffré ou sans texte extractible.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        if reader.is_encrypted:
            # Tentative avec un mot de passe vide (cas fréquent des PDF "protégés")
            if reader.decrypt("") == 0:
                raise PDFParsingError("Le PDF est protégé par un mot de passe.")

        pages_text: list[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages_text.append(text)

    except PdfReadError as exc:
        raise PDFParsingError(f"PDF illisible ou corrompu : {exc}") from exc

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise PDFParsingError(
            "Aucun texte extractible : le PDF est probablement scanné (OCR requis)."
        )
    return full_text