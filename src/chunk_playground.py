"""PDF text extraction and chunking helpers with logging for quick experiments."""

import logging
from pathlib import Path

from pypdf import PdfReader

# Keep logging consistent with other modules so experiments emit the same detail.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Return all text from a PDF as one big string."""
    logger.info("Opening PDF for extraction: %s", pdf_path)

    reader = PdfReader(str(pdf_path))  # Create a PdfReader for the provided path
    texts = []  # Container to collect text from each page

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""  # Attempt to extract text; default to empty if None
        except Exception as exc:
            logger.exception("Error extracting text from page %s of %s", i, pdf_path.name)
            text = ""  # If extraction fails for a page, fall back to an empty string
        texts.append(text)  # Store the extracted (or empty) text for this page

    joined_text = "\n".join(texts)  # Join all page texts with newlines to form one large string
    logger.info("Extracted %s characters from %s", len(joined_text), pdf_path.name)
    return joined_text


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Simple character-based chunking with overlap."""
    logger.info("Chunking text (length=%s) with chunk_size=%s, overlap=%s", len(text), chunk_size, overlap)

    chunks = []  # List to hold the resulting text chunks
    start = 0  # Starting index for the current chunk window
    n = len(text)  # Total number of characters in the input text

    while start < n:
        end = start + chunk_size  # Compute the end index for this chunk
        chunk = text[start:end]  # Slice out the current chunk of text
        if chunk.strip():  # Only add non-empty (non-whitespace-only) chunks
            chunks.append(chunk)
        start += chunk_size - overlap  # Move the start forward by chunk_size minus overlap to create overlap

    logger.info("Created %s chunks from provided text", len(chunks))
    return chunks
