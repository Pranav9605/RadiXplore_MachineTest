import fitz  
import pdfplumber
import logging
from typing import List, Tuple

logger = logging.getLogger("pdf_parser")

def parse_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, text) tuples.
    Tries PyMuPDF first, with pdfplumber as a fallback.
    """
    pages = []

    try:
        logger.info(f"[PDF] Parsing with PyMuPDF: {file_path}")
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
        doc.close()
        logger.debug(f"[PDF] PyMuPDF extracted {len(pages)} pages")
        return pages
    except Exception as e:
        logger.warning(f"[PDF] PyMuPDF failed for {file_path}: {e}")

    try:
        logger.info(f"[PDF] Falling back to pdfplumber: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            for i, pg in enumerate(pdf.pages):
                text = pg.extract_text()
                if text and text.strip():
                    pages.append((i + 1, text.strip()))
        logger.debug(f"[PDF] pdfplumber extracted {len(pages)} pages")
    except Exception as e:
        logger.error(f"[PDF] pdfplumber failed for {file_path}: {e}")

    return pages
