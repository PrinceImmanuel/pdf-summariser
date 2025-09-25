"""
app.py - Streamlit entrypoint for PDF Chatbot & Summarizer (session-only)

Responsibilities:
- Upload PDF -> extract text -> chunk -> embeddings (all kept in session state)
- Map-reduce (chunk summarization + aggregate) full-document summary
- RAG-style chat: retrieve top-k chunks, provide context-limited answers
- UI: theme toggle (dark/light), mobile keyboard auto-scroll, summary in main area
"""

# Standard library
import io
import os

# Third-party
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Local modules (project)
import config
from pdf_utils import extract_text_from_pdf_filelike, sentence_chunking_with_overlap
from embeddings import get_embeddings_for_texts
from rag import find_top_k_indices
from summarizer import summarize_chunk, aggregate_summaries

# -----------------------
# Environment + OpenAI
# -----------------------
# Load .env (if present) into os.environ before creating OpenAI clients
load_dotenv(find_dotenv())

# Create a top-level OpenAI client (submodules may also create their own)
# This client is used for the single query-embedding call in the RAG flow.
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    # Don't throw here; instead show friendly explanation in the UI later.
    _client = None
else:
    _client = OpenAI(api_key=_OPENAI_API_KEY)

# -----------------------
# Page config (once)
# -----------------------
st.set_page_config(
    page_title="PDF Chatbot & Summarizer",
    layout="wide",
    initial_sidebar_state="auto",
)

# App title
st.title("PDF Chatbot & Summarizer — Modular Session-only")

# -----------------------
# Theme toggle (light/dark)
# -----------------------
# Initialize persistent session-state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default theme

def toggle_theme():
    """Flip theme in session state."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Render a minimal header row and put the toggle on the right
header_cols = st.columns([1, 0.08])
with header_cols[0]:
    # Keep empty (title already displayed above). This column reserves space.
    pass

with header_cols[1]:
    # Use a unique key for this widget so it won't clash with other widgets.
    # The button label shows a moon when dark, sun when light.
    if st.button("🌙" if st.session_state.theme == "dark" else "☀️", key="theme_toggle_btn_top"):
        toggle_theme()

# Inject theme-aware CSS (applies across the page). Keep it simple and robust.
if st.session_state.theme == "dark":
    components.html(
        """
        <style>
        /* Dark theme */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg,#05060a,#071329) !important;
            color: #e6eef8 !important;
        }
        .summary-card { background: rgba(255,255,255,0.03); padding: 16px; border-radius: 12px; color: #dbeafe; }
        </style>
        """,
        height=0,
    )
else:
    components.html(
        """
        <style>
        /* Light theme */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg,#ffffff,#f7fafc) !important;
            color: #0b1220 !important;
        }
        .summary-card { background: #f3f6ff; padding: 16px; border-radius: 12px; color: #0b1726; }
        </style>
        """,
        height=0,
    )

# -----------------------
# Mobile keyboard helper JS
# -----------------------
# Inject a tiny script that ensures focused inputs are scrolled into view on mobile.
components.html(
    """
    <script>
    (function() {
      function scrollIntoViewDelayed(el) {
        if (!el) return;
        setTimeout(function() {
          try { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
          catch(e) { el.scrollIntoView(); }
        }, 250);
      }
      document.addEventListener('focusin', function(e) {
        const t = e.target;
        if (!t) return;
        const tag = (t.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || t.isContentEditable) {
          scrollIntoViewDelayed(t);
        }
      });
      let rTimer;
      window.addEventListener('resize', function() {
        if (rTimer) clearTimeout(rTimer);
        rTimer = setTimeout(function() {
          const active = document.activeElement;
          if (active && (active.tagName || '').toLowerCase() === 'input') {
            scrollIntoViewDelayed(active);
          }
        }, 200);
      });
      // ensure bottom safe padding
      const style = document.createElement('style');
      style.innerHTML = "body{padding-bottom:env(safe-area-inset-bottom,20px);} input, textarea { z-index:9999; position:relative; }";
      document.head.appendChild(style);
    })();
    </script>
    """,
    height=0,
)

# -----------------------
# Sidebar: upload and controls
# -----------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a single PDF", type=["pdf"])
st.sidebar.markdown("---")
st.sidebar.write("Note: embeddings and summaries are stored only in your session state (session-only).")

# -----------------------
# Session storage initialization
# -----------------------
if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "doc_text": None,
        "chunks": [],
        "embeddings": [],
        "messages": [{"role": "assistant", "content": "Upload a PDF to begin. I keep everything only in this session."}],
        "summary": None,
    }

# -----------------------
# PDF processing: extract -> chunk -> embed
# -----------------------
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    # Only process if we haven't already processed this session's PDF
    if not st.session_state.session_data["doc_text"]:
        # Extract text
        with st.spinner("Extracting PDF text..."):
            try:
                text = extract_text_from_pdf_filelike(io.BytesIO(pdf_bytes))
                if not text or not text.strip():
                    st.error("No extractable text found (PDF might be images/scanned or encrypted).")
                    st.stop()
                st.session_state.session_data["doc_text"] = text
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
                st.stop()

        # Chunk text safely (chunk_size bounds and defaults)
        with st.spinner("Chunking document..."):
            # Use session UI values if present (you may hide these controls), otherwise fall back to config
            chunk_size = int(st.session_state.get("ui_chunk_size", config.CHUNK_SIZE))
            chunk_overlap = int(st.session_state.get("ui_chunk_overlap", config.CHUNK_OVERLAP))
            # enforce sane bounds
            chunk_size = max(200, min(chunk_size, 5000))
            chunk_overlap = max(0, min(chunk_overlap, 2000))

            chunks = sentence_chunking_with_overlap(
                st.session_state.session_data["doc_text"],
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
            if not chunks:
                st.error("Chunking produced no chunks — aborting.")
                st.stop()
            st.session_state.session_data["chunks"] = chunks

        # Embeddings (in-session only)
        with st.spinner("Generating embeddings (in-session, batched)..."):
            try:
                embeddings = get_embeddings_for_texts(
                    st.session_state.session_data["chunks"],
                    model=config.EMBEDDING_MODEL,
                    batch_size=config.EMBEDDING_BATCH_SIZE,
                )
                # store as numpy arrays for faster math during retrieval
                st.session_state.session_data["embeddings"] = [np.array(e, dtype=float) for e in embeddings]
            except Exception as e:
                st.error(f"Embedding generation failed: {e}")
                st.stop()

        st.success("PDF processed into session memory. Ask a question or generate a summary.")
        st.session_state.session_data["messages"].append({"role": "assistant", "content": "PDF is ready in this session. How can I help?"})

# -----------------------
# Summary display (main area, above chat)
# -----------------------
if st.session_state.session_data.get("summary"):
    with st.expander("📄 Document Summary", expanded=False):
        # Use theme-aware CSS class "summary-card"
        st.markdown(
            f'<div class="summary-card">{st.session_state.session_data["summary"]}</div>',
            unsafe_allow_html=True,
        )

# -----------------------
# Chat history display
# -----------------------
st.subheader("Chat")
for msg in st.session_state.session_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# Summary generation (map-reduce) button (keeps using sidebar for action)
# -----------------------
if uploaded_file and st.sidebar.button("Generate full-document summary (map-reduce)"):
    if not st.session_state.session_data["chunks"]:
        st.sidebar.error("No document processed in session.")
    else:
        # Map (summarize each chunk)
        with st.spinner("Summarizing chunks (map step)..."):
            chunk_summaries = []
            # show periodic progress in sidebar
            for i, ch in enumerate(st.session_state.session_data["chunks"]):
                if i % 10 == 0:
                    st.sidebar.write(f"Summarizing chunk {i+1}/{len(st.session_state.session_data['chunks'])}...")
                chunk_summaries.append(
                    summarize_chunk(
                        ch,
                        model=config.LLM_MODEL,
                        approx_max_tokens=config.MAP_SUMMARY_TOKENS,
                    )
                )
            st.sidebar.success("Chunk summaries created — aggregating...")

        # Reduce (aggregate)
        with st.spinner("Aggregating chunk summaries (reduce step)..."):
            final_summary = aggregate_summaries(
                chunk_summaries,
                model=config.LLM_MODEL,
                max_tokens=config.FINAL_SUMMARY_TOKENS,
            )
            st.session_state.session_data["summary"] = final_summary

# -----------------------
# Chat input + RAG answering
# -----------------------
if uploaded_file and st.session_state.session_data["embeddings"]:
    user_prompt = st.chat_input("Ask a question about the PDF...")
    if user_prompt:
        # append user message to session messages (so it displays)
        st.session_state.session_data["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # assistant placeholder that will be filled with the result (with spinner)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    # require OpenAI client
                    if _client is None:
                        raise RuntimeError("OPENAI_API_KEY not found in environment. Set OPENAI_API_KEY for LLM calls.")

                    # 1) Embed the query
                    q_resp = _client.embeddings.create(input=[user_prompt], model=config.EMBEDDING_MODEL)
                    q_emb = list(map(float, q_resp.data[0].embedding))

                    # 2) Retrieve top-k indices
                    top_k = min(config.TOP_K, len(st.session_state.session_data["embeddings"]))
                    top_indices = find_top_k_indices(q_emb, st.session_state.session_data["embeddings"], top_k=top_k)

                    # 3) Build the context string from the retrieved chunks
                    # IMPORTANT: zip top_indices with chunk contents properly
                    relevant_chunks = [st.session_state.session_data["chunks"][i] for i in top_indices]
                    context = "\n\n".join([f"Chunk {i}:\n{c}" for i, c in zip(top_indices, relevant_chunks)])

                    # 4) Build a careful prompt and ask the LLM (we use summarizer's client pattern)
                    # Use summarizer module's client if available to keep consistent client config
                    try:
                        from summarizer import _client as summarizer_client
                    except Exception:
                        # fallback to top-level client
                        summarizer_client = _client

                    prompt = (
                        "You are a helpful assistant. Focus on the provided document context as your main source of truth. "
                        "If the question is broad (like 'what is this document about?'), give your best summary from the context you see. "
                        "If specific information is not present in the context, clearly say you don't have enough information.\n\n"
                        f"Document Context:\n{context}\n\n"
                        f"User's Question:\n{user_prompt}\n\n"
                        "Answer:"
                    )

                    # NOTE: this uses the OpenAI SDK pattern your codebase used previously.
                    # Ensure summarizer_client supports chat completions the same way your summarizer module expects.
                    resp = summarizer_client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that only uses the provided context."},
                            {"role": "user", "content": prompt},
                        ],
                    )

                    # Extract the assistant answer robustly
                    # Different SDK versions might return different shapes; the 'choices' path is common.
                    try:
                        answer = resp.choices[0].message.content.strip()
                    except Exception:
                        # fallback for other response shapes
                        answer = getattr(resp, "content", str(resp)).strip()

                    # Show & store
                    st.markdown(answer)
                    st.session_state.session_data["messages"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    err_txt = f"(RAG answer error: {e})"
                    st.error(err_txt)
                    st.session_state.session_data["messages"].append({"role": "assistant", "content": err_txt})

# -----------------------
# Info tip when nothing uploaded
# -----------------------
if not uploaded_file:
    st.info("Upload a PDF from the sidebar. This app stores everything only in session memory.")

# End of file
