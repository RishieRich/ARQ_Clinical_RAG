"""Core Retrieval-Augmented Generation helpers for the clinical RAG app.

This module owns the end-to-end RAG workflow:
  - reconnecting to the persisted Chroma collection
  - retrieving the top-k semantic matches for a question
  - formatting those chunks into a context block
  - invoking the local Ollama chat endpoint with the constructed prompt

Logging is enabled throughout to surface retrieval counts, context sizes, and
LLM call status for easier debugging inside Streamlit and CLI entrypoints.
"""

from pathlib import Path
import logging

import chromadb
from chromadb.utils import embedding_functions
from ollama import chat  # pip install ollama

# Consistent logging format for timestamps + module names.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Paths and model/collection settings that other modules rely on.
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"

EMBED_MODEL_NAME = "nomic-embed-text"
DEFAULT_LLM_MODEL = "deepseek-r1"  # change if you prefer another ollama model


def get_collection():
    """Reconnect to the existing Chroma collection with Ollama embeddings."""
    logger.info("Connecting to Chroma at %s for collection '%s'", CHROMA_DIR, COLLECTION_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        url="http://localhost:11434",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
    )
    logger.info("Chroma collection ready; current count: %s", collection.count())
    return collection


def retrieve_context(collection, query: str, k: int = 5):
    """Run semantic search and return top-k docs + metadata."""
    logger.info("Running retrieval for query='%s' with top_k=%s", query, k)

    result = collection.query(
        query_texts=[query],
        n_results=k,
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    logger.info("Retrieved %s documents from Chroma", len(docs))
    return docs, metas


def build_context_block(docs, metas) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    logger.info("Building context block for %s chunks", len(docs))

    blocks = []
    for doc, meta in zip(docs, metas):
        src = meta.get("source")
        idx = meta.get("chunk_index")
        header = f"[Source: {src} | chunk {idx}]"
        blocks.append(f"{header}\n{doc}")

    logger.info("Context block assembled with %s characters", sum(len(b) for b in blocks))
    return "\n\n".join(blocks)


def answer_question(
    query: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
) -> str:
    """
    Full RAG flow:
      1) retrieve top-k chunks from Chroma
      2) build a context prompt
      3) call Ollama chat model
      4) return answer text
    """
    collection = get_collection()
    docs, metas = retrieve_context(collection, query, k=top_k)

    if not docs:
        logger.warning("No context retrieved for query='%s'", query)
        return "I couldn't retrieve any relevant context for this question."

    context = build_context_block(docs, metas)
    logger.info(
        "Calling LLM model '%s' with context length=%s and top_k=%s",
        llm_model,
        len(context),
        top_k,
    )

    system_prompt = (
        "You are a clinical-trials assistant. "
        "Answer the user's question using ONLY the context given. "
        "If the answer is not in the context, say explicitly: "
        "'The answer is not available in the provided guidelines.' "
        "Cite guideline names or sections when possible, but do not invent facts."
    )

    user_prompt = (
        f"Context from clinical guidelines:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely in a few paragraphs."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat(model=llm_model, messages=messages)
        logger.info("LLM response received successfully for query='%s'", query)
    except Exception:
        logger.exception("LLM call failed for query='%s'", query)
        raise

    # Depending on ollama-python version, response may be dict or object
    # Try dict-style first, then attribute-style.
    try:
        return resp["message"]["content"]
    except (TypeError, KeyError):
        return resp.message.content


def _demo():
    """Quick CLI demo to test RAG core."""
    logger.info("RAG core demo starting...")
    q = "What is an estimand according to ICH E9(R1)?"
    logger.info("Demo question: %s", q)
    ans = answer_question(q)
    logger.info("Demo answer:\n%s", ans)


if __name__ == "__main__":
    _demo()
