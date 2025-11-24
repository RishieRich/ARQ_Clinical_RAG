from pathlib import Path
from pypdf import PdfReader

# Adjust these two if your structure changes later
BASE_DIR = Path(__file__).resolve().parents[1]       # .../clinical_rag
PDF_DIR = BASE_DIR / "data" / "pdfs"

SAMPLE_LEN = 800  # how many characters to show as a sample


def inspect_pdf(pdf_path: Path):
    print("=" * 80)
    print(f"📄 File: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    print(f"  Pages: {num_pages}")

    all_text_parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  !! Error reading page {i}: {e}")
            text = ""
        all_text_parts.append(text)

    full_text = "\n".join(all_text_parts)
    num_chars = len(full_text)
    print(f"  Characters extracted: {num_chars}")

    # show a sample snippet so you can judge quality
    snippet = full_text[:SAMPLE_LEN]
    print("\n  --- Sample text start ---")
    print(snippet.replace("\n", "⏎\n"))  # mark line breaks so you can see structure
    print("  --- Sample text end ---\n")

    return full_text


def main():
    if not PDF_DIR.exists():
        print(f"PDF directory not found: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR}\n")

    for pdf_path in pdf_files:
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()
