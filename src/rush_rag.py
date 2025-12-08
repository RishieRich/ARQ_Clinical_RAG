"""
rush_rag.py  (single-file LangChain + Chroma + Ollama RAG demo, no old chains)

Flow:
1. Define some clinical knowledge text in a Python string.
2. Wrap it in a LangChain Document.
3. Split into overlapping chunks with RecursiveCharacterTextSplitter.
4. Embed chunks with OllamaEmbeddings and store in Chroma (local vector DB).
5. For each question:
   - Retrieve top-k similar chunks from Chroma.
   - Build a prompt with those chunks as context.
   - Call Ollama LLM directly and show answer + sources.

Run from: clinical_rag/src
    (.venv) D:\...\clinical_rag\src> python rush_rag.py
"""

from pathlib import Path
import logging
from typing import List, Tuple

# ---- LangChain core & modular packages ----
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ==============================
# 1. LOGGING CONFIG
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rush_rag")

# ==============================
# 2. PATHS & MODEL NAMES
# ==============================

# .../clinical_rag
BASE_DIR = Path(__file__).resolve().parents[1]

# Chroma will persist its SQLite + index files here:
CHROMA_DIR = BASE_DIR / "data" / "langchain_simple_chroma"

# Ollama models (must exist locally; use `ollama pull` if needed)
EMBED_MODEL = "nomic-embed-text"  # embedding model
CHAT_MODEL = "llama3"             # chat model (or "deepseek-r1" etc.)

# ==============================
# 3. KNOWLEDGE TEXT AS DOCUMENTS
# ==============================

knowledge_text = """
ICH E9(R1) focuses on the statistical principles of clinical trials. It introduces the concept
of the estimand framework, which connects the trial objective with the trial analysis.
An estimand is defined by four key attributes: the population, the variable (endpoint),
the handling of intercurrent events, and the summary measure.

ICH E6 (Good Clinical Practice) provides guidance on responsibilities of sponsors,
investigators, and monitors to ensure patient safety and data integrity in clinical trials.

In oncology, endpoints such as Overall Survival (OS), Progression-Free Survival (PFS),
and Objective Response Rate (ORR) are commonly used to assess treatment benefit.
"""

# LangChain expects Documents, not raw strings
docs: List[Document] = [
    Document(
        page_content=knowledge_text,
        metadata={"source": "toy_clinical_knowledge"},
    )
]

# ==============================
# 4. CHUNKING LOGIC
# ==============================

def make_chunks(
    documents: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    Split long documents into overlapping smaller Documents.

    - chunk_size:   max characters in one chunk.
    - chunk_overlap:characters repeated between neighbors for context continuity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d docs", len(chunks), len(documents))
    return chunks

# ==============================
# 5. BUILD / LOAD CHROMA VECTOR STORE
# ==============================

def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Create (or overwrite) a Chroma vector store from the given chunks.

    Steps:
    - Ensure CHROMA_DIR exists.
    - Create OllamaEmbeddings object (talks to Ollama embedding model).
    - Use Chroma.from_documents() to:
        * embed each chunk into a vector
        * store vectors + metadata in CHROMA_DIR on disk.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedding = OllamaEmbeddings(model=EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
    )

    logger.info("Vector store built with %d chunk-documents", len(chunks))
    return vectordb

# ==============================
# 6. QUESTION ANSWERING (MANUAL RAG LOOP)
# ==============================

def answer_question(
    question: str,
    vectordb: Chroma,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """
    Manual RAG flow (no langchain.chains):

    1. Retrieve top-k relevant chunks from Chroma.
    2. Concatenate them into a context string.
    3. Build a prompt injecting {context} + {question}.
    4. Call Ollama LLM directly with that prompt.
    5. Return (answer, chunks_used).
    """
    # --- 1. Retrieve relevant chunks from Chroma ---
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    source_docs: List[Document] = retriever.get_relevant_documents(question)

    if not source_docs:
        return "I could not find relevant context for this question.", []

    # --- 2. Build context string ---
    context_parts = []
    for i, doc in enumerate(source_docs, start=1):
        context_parts.append(f"[{i}] {doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    # --- 3. Build prompt for the LLM ---
    prompt = f"""
You are a helpful clinical trial assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I don't know based on the loaded guidelines."

Context:
{context}

Question: {question}

Answer in 3–5 concise sentences:
""".strip()

    # --- 4. Call Ollama LLM directly ---
    llm = Ollama(model=CHAT_MODEL)

    # Newer LangChain returns a BaseMessage or plain str depending on integration.
    raw_response = llm.invoke(prompt)

    # Normalise to text
    if hasattr(raw_response, "content"):
        answer_text = raw_response.content
    else:
        answer_text = str(raw_response)

    return answer_text.strip(), source_docs

# ==============================
# 7. SIMPLE CLI LOOP
# ==============================

def qa_cli(vectordb: Chroma) -> None:
    """
    Terminal loop:
    - user types question
    - we run answer_question()
    - show answer + which chunks were used
    """
    print(f"RAG ready. CHAT_MODEL={CHAT_MODEL}, EMBED_MODEL={EMBED_MODEL}")
    print("Ask something about estimands / GCP / oncology endpoints.")
    print("Type 'q' to quit.")

    while True:
        question = input("\nQ: ").strip()
        if not question or question.lower() in {"q", "quit", "exit"}:
            break

        print("\nThinking...")
        answer, sources = answer_question(question, vectordb, k=4)

        print("\nA:", answer)
        print("\nSources used:")
        for i, doc in enumerate(sources, start=1):
            print(f"  [{i}] {doc.metadata.get('source', 'unknown')}")

        print("-" * 60)

# ==============================
# 8. MAIN
# ==============================

if __name__ == "__main__":
    # 1) Text -> chunks
    chunks = make_chunks(docs)

    # 2) Chunks -> Chroma vector store
    vectordb = build_vectorstore(chunks)

    # 3) Start Q&A loop
    qa_cli(vectordb)
