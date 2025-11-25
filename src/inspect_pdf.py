"""Quick PDF inspection helper for sanity-checking extraction quality.

Run this module to loop over `data/pdfs`, print basic stats for each file
(name, page count, character count), and show a short text sample. This is
intended for manual debugging, not production ingestion.
"""

from pathlib import Path  # Standard library path utility for clean path handling
from pypdf import PdfReader  # Third-party reader used to open and parse PDFs

# Adjust these two if your structure changes later
BASE_DIR = Path(__file__).resolve().parents[1]       # .../clinical_rag
PDF_DIR = BASE_DIR / "data" / "pdfs"

SAMPLE_LEN = 800  # how many characters to show as a sample


def inspect_pdf(pdf_path: Path):
    """Print summary info and a text snippet for a single PDF file."""
    print("=" * 80)  # Divider to separate output per PDF
    print(f"  File: {pdf_path.name}")  # Show which PDF is being processed

    reader = PdfReader(str(pdf_path))  # Load the PDF into a reader object
    num_pages = len(reader.pages)  # Count pages inside the PDF
    print(f"  Pages: {num_pages}")  # Report the page count

    all_text_parts = []  # Accumulates extracted text from each page
    for i, page in enumerate(reader.pages):  # Iterate through all pages by index
        try:
            text = page.extract_text() or ""  # Extract text; fallback to empty string
        except Exception as e:
            print(f"  !! Error reading page {i}: {e}")  # Log any extraction errors
            text = ""  # Keep the loop moving even if a page fails
        all_text_parts.append(text)  # Store the text segment for later joining

    full_text = "\n".join(all_text_parts)  # Combine page texts into one string
    num_chars = len(full_text)  # Count how many characters were extracted
    print(f"  Characters extracted: {num_chars}")  # Show character count summary

    # show a sample snippet so you can judge quality
    snippet = full_text[:SAMPLE_LEN]  # Grab the first N characters for preview
    print("\n  --- Sample text start ---")  # Header before the preview snippet
    print(snippet.replace("\n", "\\n\n"))  # Replace newlines with a visible marker
    print("  --- Sample text end ---\n")  # Footer after the preview snippet

    return full_text  # Return the full extracted text for potential reuse


def main():
    """Locate PDFs in the configured directory and inspect them one by one."""
    if not PDF_DIR.exists():  # Guard against missing PDF directory
        print(f"PDF directory not found: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))  # Find all PDF files, sorted
    if not pdf_files:  # If no files are present, exit early
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR}\n")  # High-level summary

    for pdf_path in pdf_files:  # Loop through each discovered PDF
        inspect_pdf(pdf_path)  # Run the inspection on the current file


if __name__ == "__main__":  # Only execute when run as a script, not imported
    main()
