"""Quick PDF inspection helper for sanity-checking extraction quality."""

import logging
from pathlib import Path  # Standard library path utility for clean path handling

from pypdf import PdfReader  # Third-party reader used to open and parse PDFs

# Configure logging to capture inspection details.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Adjust these two if your structure changes later
BASE_DIR = Path(__file__).resolve().parents[1]       # .../clinical_rag
PDF_DIR = BASE_DIR / "data" / "pdfs"

SAMPLE_LEN = 800  # how many characters to show as a sample


def inspect_pdf(pdf_path: Path):
    """Print summary info and a text snippet for a single PDF file."""
    logger.info("=" * 80)
    logger.info("File: %s", pdf_path.name)

    reader = PdfReader(str(pdf_path))  # Load the PDF into a reader object
    num_pages = len(reader.pages)  # Count pages inside the PDF
    logger.info("Pages: %s", num_pages)

    all_text_parts = []  # Accumulates extracted text from each page
    for i, page in enumerate(reader.pages):  # Iterate through all pages by index
        try:
            text = page.extract_text() or ""  # Extract text; fallback to empty string
        except Exception as e:
            logger.exception("Error reading page %s of %s", i, pdf_path.name)
            text = ""  # Keep the loop moving even if a page fails
        all_text_parts.append(text)  # Store the text segment for later joining

    full_text = "\n".join(all_text_parts)  # Combine page texts into one string
    num_chars = len(full_text)  # Count how many characters were extracted
    logger.info("Characters extracted: %s", num_chars)

    snippet = full_text[:SAMPLE_LEN]  # Grab the first N characters for preview
    logger.info("--- Sample text start ---")
    logger.info("%s", snippet.replace("\n", "\\n\n"))  # Replace newlines with a visible marker
    logger.info("--- Sample text end ---")

    return full_text  # Return the full extracted text for potential reuse


def main():
    """Locate PDFs in the configured directory and inspect them one by one."""
    if not PDF_DIR.exists():  # Guard against missing PDF directory
        logger.error("PDF directory not found: %s", PDF_DIR)
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))  # Find all PDF files, sorted
    if not pdf_files:  # If no files are present, exit early
        logger.warning("No PDFs found in %s", PDF_DIR)
        return

    logger.info("Found %s PDF(s) in %s", len(pdf_files), PDF_DIR)

    for pdf_path in pdf_files:  # Loop through each discovered PDF
        inspect_pdf(pdf_path)  # Run the inspection on the current file


if __name__ == "__main__":  # Only execute when run as a script, not imported
    main()
