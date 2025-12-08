# Clinical RAG Copilot

A local-first Retrieval Augmented Generation (RAG) playground for clinical guidelines. The project ingests guideline PDFs into a Chroma vector store (embedded with Ollama), then serves a Streamlit chat UI and small CLI utilities for debugging retrieval quality.

## Repository structure

- `data/pdfs/` – Source documents: ICH E6 (GCP), ICH E9(R1), and FDA oncology endpoint guidance PDFs that feed the RAG pipeline.
- `data/chroma_db/` – Persistent Chroma database populated by the ingestion script.
- `src/app.py` – Streamlit front-end for chatting with the Clinical RAG Copilot (select Ollama model, set top-k, view responses and latency).
- `src/rag_core.py` – Core RAG workflow: reconnects to the Chroma collection, retrieves top-k chunks, builds the context block, and calls the Ollama chat endpoint.
- `src/ingest.py` – PDF ingestion pipeline: extracts text, chunks it, and writes documents plus metadata into the Chroma collection using Ollama embeddings.
- `src/chunk_playground.py` – Helpers for PDF text extraction and simple overlapping character chunking.
- `src/text_utils.py` – Shared utility wrapper around the chunking helpers with convenience logging and demo chunking configs.
- `src/inspect_pdf.py` – Quick PDF inspection script to sanity-check extraction quality and length.
- `src/retriever_playground.py` – CLI loop to issue retrieval queries and log the ranked chunks returned from Chroma.
- `src/rush_rag.py` – Early stub for an alternative pipeline (currently only sets up paths and imports).

## Prerequisites

- Python environment with the dependencies in `requirements.txt` installed (Streamlit, Chroma, Ollama client, PyPDF, etc.).
- An Ollama server running locally with the embedding model `nomic-embed-text` and a chat-capable model (defaults to `deepseek-r1` but configurable in the UI).

## Ingesting the guideline corpus

1. Place additional guideline PDFs into `data/pdfs/` if needed.
2. Run the ingestion script to populate or refresh the Chroma collection:
   - From the repository root: `cd src && python ingest.py`
   - The script extracts text, chunks it (default 1200 chars with 200 overlap), and persists documents plus metadata into `data/chroma_db/`.

## Running the Streamlit UI

1. Ensure the Chroma database is populated (see ingestion step) and Ollama is running.
2. Start the app from the repository root: `cd src && streamlit run app.py`
3. Use the sidebar to choose the Ollama model and retrieval depth (top-k). The chat history is preserved per session, and response latency is displayed beneath each answer.

## Debugging and experimentation utilities

- **Inspect PDF extraction:** `cd src && python inspect_pdf.py` to view page counts, extracted characters, and sample snippets for each PDF.
- **Tune chunking:** `cd src && python text_utils.py` to log chunk counts and sample chunks across a few chunk size/overlap configurations.
- **Probe retrieval quality:** `cd src && python retriever_playground.py` to issue ad-hoc questions and review the ranked chunks with their source filenames and indices.

## Notes

- Logging is enabled across scripts with a common format to make retrieval counts, chunk sizes, and LLM call status easy to trace.
- The repository is organized to keep data (PDFs and vector store) under `data/` while the application and utilities live in `src/` for direct CLI execution.
