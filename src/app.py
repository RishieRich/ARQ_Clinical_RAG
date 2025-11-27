"""Streamlit front-end for the Clinical RAG Copilot with logging instrumentation."""

import logging

import streamlit as st
from rag_core import answer_question, DEFAULT_LLM_MODEL

# Configure logging early so Streamlit reruns keep a consistent format.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Page config
# -------------------------------------------------
logger.info("Loading Streamlit UI layout configuration")
st.set_page_config(
    page_title="Clinical RAG Copilot",
    page_icon=":speech_balloon:",
    layout="wide",
)

# -------------------------------------------------
# Custom CSS for layout & styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1f2937 0, #020617 40%, #020617 100%);
    }

    /* Center main content and limit width */
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        margin: 0 auto;
    }

    /* Header card */
    .title-card {
        background: linear-gradient(135deg, #4f46e5 0%, #22c55e 40%, #0ea5e9 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 1.4rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        color: #f9fafb;
    }

    .title-main {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .title-sub {
        font-size: 0.95rem;
        opacity: 0.95;
    }

    /* Pills on the right of header */
    .metric-pill {
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.75rem;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.45);
        color: #e5e7eb;
        margin-right: 0.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1f2933;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 1.2rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        background-color: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Chat input bar */
    .stChatInput textarea {
        border-radius: 999px !important;
        border: 1px solid #4f46e5 !important;
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }

    /* Placeholder text color in chat input */
    .stChatInput textarea::placeholder {
        color: #64748b !important;
    }

    /* Remove top padding above first chat message */
    div[data-testid="stVerticalBlock"] > div:first-child[data-testid="stChatMessage"] {
        margin-top: 0.5rem;
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
    "Top-k chunks",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Number of retrieved chunks from Chroma to pass into the LLM context.",
)
logger.info("Sidebar top_k set to %s", top_k)

st.sidebar.markdown("---")
st.sidebar.caption("Backend: Chroma + Ollama embeddings (nomic-embed-text)")

# -------------------------------------------------
# Header (main area)
# -------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(
        """
        <div class="title-card">
          <div class="title-main">Clinical RAG Copilot</div>
          <div class="title-sub">
            Ask questions about ICH E6 (GCP), ICH E9(R1), and FDA oncology endpoints.<br/>
            Answers are grounded in your local guideline corpus using Chroma + Ollama.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    st.markdown(
        """
        <div>
          <span class="metric-pill">3 guidelines indexed</span>
          <span class="metric-pill">Vector search via Chroma</span>
          <span class="metric-pill">Local LLM via Ollama</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # spacer

# -------------------------------------------------
# Chat history
# -------------------------------------------------
if "history" not in st.session_state:
    # each item: {"role": "user" | "assistant", "content": str}
    st.session_state.history = []
    logger.info("Initialized new chat history in session state")
else:
    logger.info("Restored chat history with %s messages", len(st.session_state.history))

# Show existing history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# If no history, show a small hint in the middle
if not st.session_state.history:
    st.markdown(
        "<p style='color:#9ca3af; font-size:0.9rem; margin-top:0.8rem;'>"
        "Tip: start with something like "
        "<em>What is an estimand according to ICH E9(R1)?</em>"
        "</p>",
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

    # 2) Generate assistant response via RAG
    with st.chat_message("assistant"):
        with st.spinner("Reasoning over guidelines..."):
            try:
                answer = answer_question(
                    user_input,
                    llm_model=llm_model,
                    top_k=top_k,
                )
                logger.info("Answer generated (chars=%s)", len(answer))
            except Exception as e:
                logger.exception("Error while generating answer for user input")
                answer = f"Error while generating answer: `{e}`"

        st.markdown(answer)

    # 3) Store assistant message
    st.session_state.history.append({"role": "assistant", "content": answer})
    logger.info("Chat history updated; total messages=%s", len(st.session_state.history))
