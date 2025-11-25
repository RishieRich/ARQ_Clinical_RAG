# Standard library path helper for resolving project directories
from pathlib import Path
# Import PDF text extraction and chunking utilities from the sibling module
from chunk_playground import extract_text_from_pdf, chunk_text

# Resolve the repository root (.../clinical_rag) from this file's location
BASE_DIR = Path(__file__).resolve().parents[1]   # .../clinical_rag
# Point to the folder containing PDF files used for experimentation
PDF_DIR = BASE_DIR / "data" / "pdfs"


def show_chunks_for_config(pdf_path: Path, chunk_size: int, overlap: int):
    # Print a separator for readability in the console output
    print("=" * 80)
    # Indicate which PDF is being processed
    print(f"File: {pdf_path.name}")
    # Display the chunking configuration parameters being tested
    print(f"Config: chunk_size={chunk_size}, overlap={overlap}")

    # Extract the full text from the provided PDF
    text = extract_text_from_pdf(pdf_path)
    # Break the text into overlapping chunks using the provided configuration
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Report total character count for the extracted text
    print(f"Total characters: {len(text)}")
    # Report how many chunks were produced
    print(f"Number of chunks: {len(chunks)}")
    # If no chunks were created, exit early
    if not chunks:
        return

    # Show the first two chunks to inspect boundary behavior
    for i, ch in enumerate(chunks[:2]):
        # Separator between displayed chunks
        print("-" * 40)
        # Identify the chunk index and its length
        print(f"Chunk {i} (len={len(ch)}):")
        # Print the first 600 characters, with visible newlines
        print(ch[:600].replace("\n", "\\n\n"))
        # Indicate that the chunk preview is truncated
        print("\n[...]")
    # Add an extra blank line after processing
    print()


def main():
    # Collect all PDF files in the target directory, sorted for consistency
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    # If no PDFs are found, notify the user and exit
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    # Choose the first PDF as the default target (adjust as needed)
    target_pdf = pdf_files[0]  # or choose by name/index

    # Define several chunking configurations to experiment with
    configs = [
        (800, 200),
        (1200, 200),
        (1600, 300),
    ]

    # Run the chunk display helper for each configuration
    for cs, ov in configs:
        show_chunks_for_config(target_pdf, cs, ov)


if __name__ == "__main__":
    # Execute the script when run directly from the command line
    main()
