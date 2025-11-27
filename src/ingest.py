"""Ingestion script that chunks PDFs and loads them into Chroma with logging."""

import logging
from pathlib import Path

import chromadb  # Vector database client used for persistence and search
from chromadb.utils import embedding_functions  # Helpers for embedding backends

# Reuse the PDF extraction and chunking helpers from the shared utils
from text_utils import extract_text_from_pdf, chunk_text

# Consistent logging so CLI runs emit the same detail.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Resolve the project root (.../clinical_rag) relative to this file
BASE_DIR = Path(__file__).resolve().parents[1]
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
    """Create a Chroma client and collection configured with Ollama embeddings."""
    logger.info("Connecting to Chroma at %s", CHROMA_DIR)
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
    logger.info("Collection '%s' ready with %s documents", COLLECTION_NAME, collection.count())
    return client, collection


def ingest_pdfs():
    """Read all PDFs in PDF_DIR, chunk them, and add chunks to Chroma."""
    if not PDF_DIR.exists():
        logger.error("PDF directory not found: %s", PDF_DIR)
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))  # Collect all PDF files, sorted for consistent ordering
    if not pdf_files:
        logger.warning("No PDFs found in %s", PDF_DIR)
        return

    logger.info("Found %s PDF(s): %s", len(pdf_files), ", ".join(f.name for f in pdf_files))

    # Prepare Chroma client and collection for ingestion
    client, collection = build_client_and_collection()

    # Accumulators for IDs, documents, and metadata to send to Chroma
    all_ids, all_docs, all_metas = [], [], []

    # Process each PDF file one by one
    for pdf_path in pdf_files:
        logger.info("Processing: %s", pdf_path.name)

        # Extract raw text from the PDF
        text = extract_text_from_pdf(pdf_path)
        logger.info("Extracted %s characters from %s", len(text), pdf_path.name)

        # Convert the text into overlapping chunks
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        logger.info("Created %s chunks from %s", len(chunks), pdf_path.name)

        # Build IDs, documents, and metadata entries for each chunk
        for idx, chunk in enumerate(chunks):
            cid = f"{pdf_path.stem}_{idx}"  # Unique ID combines the PDF stem and chunk index
            meta = {
                "source": pdf_path.name,
                "chunk_index": idx,
            }
            all_ids.append(cid)
            all_docs.append(chunk)
            all_metas.append(meta)

    # If nothing was produced (e.g., PDFs were empty), exit gracefully
    if not all_docs:
        logger.warning("No chunks to add; exiting without modifying Chroma")
        return

    # Send all accumulated chunks to the Chroma collection (triggers embedding)
    logger.info("Adding %s chunks to Chroma (this calls Ollama for embeddings)...", len(all_docs))
    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas)

    logger.info("Ingestion complete.")
    logger.info("Collection '%s' now has %s documents.", COLLECTION_NAME, collection.count())


def main():
    """Entry point for running ingestion directly."""
    ingest_pdfs()


if __name__ == "__main__":
    main()
