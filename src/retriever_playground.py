"""CLI playground for retrieval results with verbose logging."""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"


def get_collection():
    """Reconnect to the Chroma collection with Ollama embeddings."""
    logger.info("Connecting to Chroma at %s", CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
    )
    logger.info("Collection '%s' ready with %s documents", COLLECTION_NAME, collection.count())
    return collection


def query_once(collection, question: str, k: int = 5):
    """Run a single retrieval query and log the ranked results."""
    logger.info("Question: %s", question)

    result = collection.query(
        query_texts=[question],
        n_results=k,
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    logger.info("Top %s retrieved chunks:", k)
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        logger.info("=" * 80)
        logger.info("Rank #%s", i + 1)
        logger.info("Source      : %s", meta.get("source"))
        logger.info("Chunk index : %s", meta.get("chunk_index"))
        logger.info("-" * 80)
        logger.info("%s", doc[:600].replace("\n", "\\n\n"))
        logger.info("[...]")


def main():
    """Interactive loop to issue retrieval queries."""
    logger.info("retriever_playground.py starting")

    collection = get_collection()
    logger.info("Loaded collection '%s' with %s documents.", COLLECTION_NAME, collection.count())

    while True:
        question = input("\nType a clinical question (or 'q' to quit): ").strip()
        if not question or question.lower() in {"q", "quit", "exit"}:
            logger.info("Exiting retrieval playground loop.")
            break

        query_once(collection, question, k=5)


if __name__ == "__main__":
    main()
