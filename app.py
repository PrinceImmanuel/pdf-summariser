"""
app.py
Streamlit app entrypoint. Orchestrates:
- PDF upload
- extraction -> chunking -> embeddings (session-only)
- map-reduce summarization
- chat (RAG) using top-k chunks

Everything is kept in st.session_state (SESSION_ONLY behavior).
"""

import io
import streamlit as st
import numpy as np
from openai import OpenAI
import os

# Local module imports
import config
from pdf_utils import extract_text_from_pdf_filelike, sentence_chunking_with_overlap
from embeddings import get_embeddings_for_texts
from rag import find_top_k_indices
from summarizer import summarize_chunk, aggregate_summaries

# Setup OpenAI client (embedding / chat calls are done in module functions that also use env var)
# NOTE: the submodules already create their own OpenAI client via OPENAI_API_KEY env var.
# This top-level client is not strictly necessary here, but we keep it for potential direct calls.
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())  # this loads .env file values into os.environ

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

st.set_page_config(
    page_title="PDF Chatbot & Summarizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit page config
st.set_page_config(page_title="PDF Chat & Summarizer (session-only)", layout="wide")
st.title("PDF Chatbot & Summarizer — Modular Session-only")

st.markdown(
    """
Upload a PDF and everything will be kept only in your session memory (no disk, no Pinecone).
When the session ends all data is discarded.
"""
)
# Make sidebar wider
st.markdown(
    """
    <style>
        /* Sidebar width */
        section[data-testid="stSidebar"] {
            width: 500px !important;   /* default ~250px */
        }
        /* Ensure main area shrinks instead */
        section[data-testid="stSidebar"] > div:first-child {
            width: 500px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a single PDF", type="pdf")


st.sidebar.markdown("---")
st.sidebar.write("Note: embeddings and summaries are stored only in your session state.")

# Initialize session_data container
if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "doc_text": None,
        "chunks": [],
        "embeddings": [],  # list of Python lists or numpy arrays
        "messages": [{"role": "assistant", "content": "Upload a PDF to begin. I keep everything only in this session."}],
        "summary": None,
    }

# Process uploaded file (extract -> chunk -> embeddings) only if not already processed in this session
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    if not st.session_state.session_data["doc_text"]:
        # Extract
        with st.spinner("Extracting PDF text..."):
            try:
                text = extract_text_from_pdf_filelike(io.BytesIO(pdf_bytes))
                if not text.strip():
                    st.error("No extractable text found (PDF might be scanned images or encrypted).")
                    st.stop()
                st.session_state.session_data["doc_text"] = text
            except Exception as e:
                st.error(str(e))
                st.stop()

        # Chunk
        with st.spinner("Chunking document..."):
            # safe assignment
            chunk_size = int(st.session_state.get("ui_chunk_size", config.CHUNK_SIZE))
            chunk_overlap = int(st.session_state.get("ui_chunk_overlap", config.CHUNK_OVERLAP))
            # bounding values (optional)
            chunk_size = max(200, min(chunk_size, 5000))
            chunk_overlap = max(0, min(chunk_overlap, 2000))

            chunks = sentence_chunking_with_overlap(st.session_state.session_data["doc_text"], chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                st.error("Chunking produced no chunks — aborting.")
                st.stop()
            st.session_state.session_data["chunks"] = chunks

        # Embeddings (in-session)
        with st.spinner("Generating embeddings (in-session, batched)..."):
            try:
                embeddings = get_embeddings_for_texts(st.session_state.session_data["chunks"], model=config.EMBEDDING_MODEL, batch_size=config.EMBEDDING_BATCH_SIZE)
                # store as numpy arrays in session for faster math
                st.session_state.session_data["embeddings"] = [np.array(e, dtype=float) for e in embeddings]
            except Exception as e:
                st.error(f"Embedding generation failed: {e}")
                st.stop()

        st.success("PDF processed into session memory. Ask a question or generate a summary.")
        st.session_state.session_data["messages"].append({"role": "assistant", "content": "PDF is ready in this session. How can I help?"})

# Display chat history
st.subheader("Chat")
for msg in st.session_state.session_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Summary button (map-reduce)
if uploaded_file and st.sidebar.button("Generate full-document summary (map-reduce)"):
    if not st.session_state.session_data["chunks"]:
        st.sidebar.error("No document processed in session.")
    else:
        with st.spinner("Summarizing chunks (map step)..."):
            chunk_summaries = []
            for i, ch in enumerate(st.session_state.session_data["chunks"]):
                if i % 10 == 0:
                    st.sidebar.write(f"Summarizing chunk {i+1}/{len(st.session_state.session_data['chunks'])}...")
                chunk_summaries.append(summarize_chunk(ch, model=config.LLM_MODEL, approx_max_tokens=config.MAP_SUMMARY_TOKENS))
            st.sidebar.success("Chunk summaries created — aggregating...")

        with st.spinner("Aggregating chunk summaries (reduce step)..."):
            final_summary = aggregate_summaries(chunk_summaries, model=config.LLM_MODEL, max_tokens=config.FINAL_SUMMARY_TOKENS)
            st.session_state.session_data["summary"] = final_summary

        st.sidebar.subheader("Document Summary")
        st.sidebar.info(st.session_state.session_data["summary"])

# Chat input & RAG answering
if uploaded_file and st.session_state.session_data["embeddings"]:
    user_prompt = st.chat_input("Ask a question about the PDF...")
    if user_prompt:
        st.session_state.session_data["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    # Get embedding for query (single)
                    q_resp = _client.embeddings.create(input=[user_prompt], model=config.EMBEDDING_MODEL)
                    q_emb = list(map(float, q_resp.data[0].embedding))

                    # Retrieve top-k indices using rag helper
                    top_k = min(config.TOP_K, len(st.session_state.session_data["embeddings"]))
                    top_indices = find_top_k_indices(q_emb, st.session_state.session_data["embeddings"], top_k=top_k)

                    # Build context from selected chunks
                    relevant_chunks = [st.session_state.session_data["chunks"][i] for i in top_indices]
                    context = "\n\n".join([f"Chunk {i}:\n{c}" for i, c in zip(top_indices, relevant_chunks)])

                    # Ask LLM to answer using only this context
                    # We use the summarizer module's pattern for calling the LLM, but craft a question-specific prompt
                    from summarizer import _client as summarizer_client  # re-use openai client inside summarizer if needed
                    prompt = (
                                "You are a helpful assistant. Focus on the provided document context as your main source of truth. "
                                "If the question is broad (like 'what is this document about?'), give your best summary from the context you see. "
                                "If specific information is not present in the context, clearly say you don't have enough information.\n\n"
                                f"Document Context:\n{context}\n\n"
                                f"User's Question:\n{user_prompt}\n\n"
                                "Answer:"
                            )

                    resp = summarizer_client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that only uses the provided context."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    answer = resp.choices[0].message.content.strip()

                    st.markdown(answer)
                    st.session_state.session_data["messages"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    err_txt = f"(RAG answer error: {e})"
                    st.error(err_txt)
                    st.session_state.session_data["messages"].append({"role": "assistant", "content": err_txt})

# Info tip if nothing is uploaded
if not uploaded_file:
    st.info("Upload a PDF in the sidebar. This app stores everything only in session memory.")
