# Standard library path object for file system work
from pathlib import Path
# PDF reader library used to open and parse PDF files
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    # Docstring describes the purpose of the function
    """Return all text from a PDF as one big string."""
    # Create a PdfReader for the provided path (string path avoids type issues)
    reader = PdfReader(str(pdf_path))
    # Container to collect text from each page
    texts = []
    # Loop through every page in the PDF with its index
    for i, page in enumerate(reader.pages):
        try:
            # Attempt to extract text for the current page; default to empty if None
            text = page.extract_text() or ""
        except Exception:
            # If extraction fails for a page, fall back to an empty string
            text = ""
        # Store the extracted (or empty) text for this page
        texts.append(text)
    # Join all page texts with newlines to form one large string
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    # Docstring clarifies that we are doing character-based chunking with overlap
    """Simple character-based chunking with overlap."""
    # List to hold the resulting text chunks
    chunks = []
    # Starting index for the current chunk window
    start = 0
    # Total number of characters in the input text
    n = len(text)

    # Continue creating chunks until we've covered the full text
    while start < n:
        # Compute the end index for this chunk
        end = start + chunk_size
        # Slice out the current chunk of text
        chunk = text[start:end]
        # Only add non-empty (non-whitespace-only) chunks
        if chunk.strip():
            chunks.append(chunk)
        # Move the start forward by chunk_size minus overlap to create overlap
        start += chunk_size - overlap

    # Return the list of generated chunks
    return chunks
