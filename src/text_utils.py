"""Shared text utilities for PDF experiments with detailed logging."""

import logging
from pathlib import Path

from chunk_playground import extract_text_from_pdf, chunk_text

# Align logging with the rest of the project so outputs are uniform.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Resolve the repository root (.../clinical_rag) from this file's location
BASE_DIR = Path(__file__).resolve().parents[1]
# Point to the folder containing PDF files used for experimentation
PDF_DIR = BASE_DIR / "data" / "pdfs"


def show_chunks_for_config(pdf_path: Path, chunk_size: int, overlap: int):
    """Display chunking stats and sample chunks for the given configuration."""
    logger.info("=" * 80)
    logger.info("File: %s", pdf_path.name)
    logger.info("Config: chunk_size=%s, overlap=%s", chunk_size, overlap)

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    logger.info("Total characters: %s", len(text))
    logger.info("Number of chunks: %s", len(chunks))
    if not chunks:
        logger.warning("No chunks produced for %s", pdf_path.name)
        return

    for i, ch in enumerate(chunks[:2]):
        logger.info("-" * 40)
        logger.info("Chunk %s (len=%s):", i, len(ch))
        logger.info("%s", ch[:600].replace("\n", "\\n\n"))
        logger.info("[...]")
    logger.info("")  # Blank line for readability


def main():
    """Run chunk inspection across a small set of configurations."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", PDF_DIR)
        return

    target_pdf = pdf_files[0]  # Choose the first PDF as the default target
    logger.info("Using target PDF: %s", target_pdf.name)

    configs = [
        (800, 200),
        (1200, 200),
        (1600, 300),
    ]

    for cs, ov in configs:
        show_chunks_for_config(target_pdf, cs, ov)


if __name__ == "__main__":
    main()
