# High-level ingestion script that chunks PDFs and writes them into Chroma
# with Ollama embeddings.
from pathlib import Path  # Path utility for filesystem navigation

import chromadb  # Vector database client used for persistence and search
from chromadb.utils import embedding_functions  # Helpers for embedding backends

# Reuse the PDF extraction and chunking helpers from the shared utils
from text_utils import extract_text_from_pdf, chunk_text

# Resolve the project root (.../clinical_rag) relative to this file
BASE_DIR = Path(__file__).resolve().parents[1]  # .../clinical_rag
# Folder containing the source PDF documents to ingest
PDF_DIR = BASE_DIR / "data" / "pdfs"
# Folder where the Chroma persistent database will live
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# Name for the Chroma collection that will store the ingested chunks
COLLECTION_NAME = "clinical_guidelines"

# Chunking configuration (tune based on chunk_playground experiments)
CHUNK_SIZE = 1200  # Number of characters per chunk
OVERLAP = 200  # Characters of overlap between adjacent chunks


def build_client_and_collection():
    # Create a Chroma client that persists its data under CHROMA_DIR
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Configure an embedding function that calls the local Ollama server
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434",
    )

    # Get or create the target collection, binding it to the embedding function
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
    )
    # Return both client and collection for downstream use
    return client, collection


def ingest_pdfs():
    # Ensure the PDF directory exists before attempting ingestion
    if not PDF_DIR.exists():
        print(f"!! PDF directory not found: {PDF_DIR}")
        return

    # Collect all PDF files, sorted for consistent ordering
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    # Bail out early if no PDFs are available
    if not pdf_files:
        print(f"!! No PDFs found in {PDF_DIR}")
        return

    # Announce which files will be processed
    print(f"-> Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"   - {f.name}")
    print()

    # Prepare Chroma client and collection for ingestion
    client, collection = build_client_and_collection()

    # Accumulators for IDs, documents, and metadata to send to Chroma
    all_ids, all_docs, all_metas = [], [], []

    # Process each PDF file one by one
    for pdf_path in pdf_files:
        # Visual separator for per-file output
        print("=" * 80)
        # Indicate which PDF is currently being handled
        print(f"-> Processing: {pdf_path.name}")

        # Extract raw text from the PDF
        text = extract_text_from_pdf(pdf_path)
        # Report how many characters were extracted
        print(f"   + extracted {len(text)} characters")

        # Convert the text into overlapping chunks
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        # Report how many chunks were produced
        print(f"   + created {len(chunks)} chunks")

        # Build IDs, documents, and metadata entries for each chunk
        for idx, chunk in enumerate(chunks):
            # Unique ID combines the PDF stem and chunk index
            cid = f"{pdf_path.stem}_{idx}"
            # Metadata describing the source and position of the chunk
            meta = {
                "source": pdf_path.name,
                "chunk_index": idx,
            }
            # Append the current chunk's data to the accumulators
            all_ids.append(cid)
            all_docs.append(chunk)
            all_metas.append(meta)

    # If nothing was produced (e.g., PDFs were empty), exit gracefully
    if not all_docs:
        print("!! No chunks to add")
        return

    # Send all accumulated chunks to the Chroma collection (triggers embedding)
    print("\n-> Adding chunks to Chroma (this calls Ollama for embeddings)...")
    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
    # Confirm ingestion completion
    print("✓ Ingestion complete.")

    # Report the final document count in the collection
    print(f"-> Collection '{COLLECTION_NAME}' now has {collection.count()} documents.")


def main():
    # Entry point for running ingestion directly
    ingest_pdfs()


if __name__ == "__main__":
    # Only run ingestion when the script is executed as the main module
    main()
