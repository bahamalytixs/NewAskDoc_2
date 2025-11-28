"""
🦜🔗 Ask the Doc – Streamlit + LangChain 0.1 demo
Streamlit Cloud–ready version
"""

import streamlit as st
from pathlib import Path

# ---------- LangChain 0.1‑compatible imports ----------
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
# -----------------------------------------------------

# ---------------- Streamlit page configuration ----------------
st.set_page_config(page_title="🦜🔗 Ask the Doc App", layout="centered")

# ---------------- Session‑state helpers -----------------------
def _reset_answers_if_new_file() -> None:
    """Clear stored answers and cached vectorstore when a new file is uploaded."""
    if "last_file_hash" not in st.session_state:
        st.session_state.last_file_hash = None

    if uploaded_file:
        cur_hash = hash(uploaded_file.getvalue())
        if st.session_state.last_file_hash != cur_hash:
            st.session_state.answers = []
            st.session_state.last_file_hash = cur_hash
            if "_make_store" in st.session_state:
                del st.session_state["_make_store"]

if "answers" not in st.session_state:
    st.session_state.answers = []

# ---------------- UI ----------------
st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader(
    "Upload a plain‑text article", type=["txt"], accept_multiple_files=False
)

# Use API key from Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("API key not found!")
else:
    st.write("API key loaded successfully")

if uploaded_file:
    _reset_answers_if_new_file()

query_text = st.text_input(
    "Enter your question:",
    placeholder="e.g. “Summarise the main points.”",
    disabled=not uploaded_file,
)

# ---------------- Cached vector‑store builder ----------------
@st.cache_resource(show_spinner=False)
def _make_store(_docs_tuple: tuple, api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    return Chroma.from_documents(list(_docs_tuple), embeddings)

# ---------------- QA Chain Builder ----------------
def build_qa_chain(file_bytes: bytes, api_key: str, query: str) -> str:
    raw_text = file_bytes.decode(errors="replace")
    splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=200)
    docs = splitter.create_documents([raw_text])
    store = _make_store(tuple(docs), api_key)
    retriever = store.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key),
        retriever=retriever,
        chain_type="stuff",
    )
    return qa.run(query)

# ---------------- Run QA ----------------
if uploaded_file and query_text and api_key:
    if st.button("Run QA"):
        with st.spinner("Generating answer..."):
            try:
                answer = build_qa_chain(
                    file_bytes=uploaded_file.getvalue(),
                    api_key=api_key,
                    query=query_text
                )
                st.subheader("Answer")
                st.write(answer)
                st.session_state.answers.append(answer)
            except Exception as e:
                st.error(f"Error running QA chain: {e}")
else:
    if not uploaded_file:
        st.info("Please upload a .txt file first.")
    elif not query_text:
        st.info("Please enter a question to ask.")
    elif not api_key:
        st.info("API key not set in Streamlit secrets.")

