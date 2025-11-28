"""Streamlit front-end for the Clinical RAG Copilot with logging instrumentation."""

import logging
import time  # NEW

import streamlit as st
from rag_core import answer_question, DEFAULT_LLM_MODEL

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

logger.info("Loading Streamlit UI layout configuration")

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Clinical RAG Copilot",
    page_icon=":speech_balloon:",
    layout="wide",
)

# -------------------------------------------------
# Custom CSS – clean, simple, no weird boxes
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background-color: #e5e7eb;   /* soft grey */
        color: #111827;
    }

    /* Center main content and limit width */
    .block-container {
        max-width: 1000px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }

    /* Header card */
    .header-card {
        background-color: #ffffff;
        padding: 1.2rem 1.5rem;
        border-radius: 0.9rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.9rem;
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
        color: #111827;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
    }

    /* Metrics row under header */
    .metrics-row {
        margin-top: 0.6rem;
    }
    .metric-pill {
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.78rem;
        background-color: #dbeafe;
        color: #1f2937;
        border: 1px solid #bfdbfe;
        margin-right: 0.35rem;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }

    /* Chat messages – simple cards on grey bg */
    [data-testid="stChatMessage"] {
        border-radius: 0.7rem;
        padding: 0.6rem 0.75rem;
        margin-bottom: 0.6rem;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    /* Chat input at bottom */
    .stChatInput textarea {
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        background-color: #ffffff !important;
        color: #111827 !important;
        font-size: 0.95rem !important;
    }
    .stChatInput textarea::placeholder {
        color: #9ca3af !important;
    }

    .empty-hint {
        color: #6b7280;
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar (settings)
# -------------------------------------------------
logger.info("Rendering sidebar controls for model selection and retrieval depth")
st.sidebar.header("Settings")
llm_model = st.sidebar.text_input(
    "Ollama model",
    value=DEFAULT_LLM_MODEL,  # e.g., "deepseek-r1"
    help="Chat-capable model you have pulled in Ollama "
         "(e.g. deepseek-r1, llama3.1, qwen2.5).",
)
logger.info("Sidebar model set to '%s'", llm_model)

top_k = st.sidebar.slider(
    "Top-k chunks from Chroma",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
)
logger.info("Sidebar top_k set to %s", top_k)

st.sidebar.markdown("---")
st.sidebar.caption("Backend: Chroma + Ollama embeddings (nomic-embed-text)")

# -------------------------------------------------
# Header (main area)
# -------------------------------------------------
st.markdown(
    """
    <div class="header-card">
      <div class="header-title">Clinical RAG Copilot</div>
      <div class="header-subtitle">
        Ask questions about ICH E6 (GCP), ICH E9(R1), and FDA oncology endpoints.<br/>
        Answers are grounded in your local guideline corpus using Chroma + Ollama.
      </div>
      <div class="metrics-row">
        <span class="metric-pill">📚 3 guidelines indexed</span>
        <span class="metric-pill">🔍 Vector search via Chroma</span>
        <span class="metric-pill">🤖 Local LLM via Ollama</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Chat history
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
    logger.info("Initialized new chat history in session state")
else:
    logger.info("Restored chat history with %s messages", len(st.session_state.history))

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.history:
    st.markdown(
        "<div class='empty-hint'>"
        "Tip: start with something like "
        "<em>What is an estimand according to ICH E9(R1)?</em>"
        "</div>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Chat input
# -------------------------------------------------
user_input = st.chat_input("Type your clinical question...")

if user_input:
    logger.info("Received user question (chars=%s)", len(user_input))

    # 1) Append and render user message
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Generate assistant response via RAG (measure time)
    with st.chat_message("assistant"):
        with st.spinner("Reasoning over guidelines..."):
            start = time.perf_counter()
            try:
                answer = answer_question(
                    user_input,
                    llm_model=llm_model,
                    top_k=top_k,
                )
                elapsed = time.perf_counter() - start
                logger.info(
                    "Answer generated (chars=%s) in %.2f seconds",
                    len(answer),
                    elapsed,
                )
            except Exception as e:
                logger.exception("Error while generating answer for user input")
                answer = f"Error while generating answer: `{e}`"
                elapsed = None

        st.markdown(answer)

        # Show timing info under the answer
        if elapsed is not None:
            st.caption(f"⏱️ Response time: {elapsed:.1f} seconds")

    # 3) Store assistant message
    st.session_state.history.append({"role": "assistant", "content": answer})
    logger.info("Chat history updated; total messages=%s", len(st.session_state.history))
