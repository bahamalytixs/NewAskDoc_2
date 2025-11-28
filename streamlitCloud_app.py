import streamlit as st
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="🦜🔗 Ask the Doc App", layout="centered")
st.title("🦜🔗 Ask the Doc App")

# ---------------- Session state ----------------
if "answers" not in st.session_state:
    st.session_state.answers = []

if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader("Upload a plain-text article", type=["txt"])

def _reset_answers_if_new_file():
    """Clear stored answers and cached vectorstore when a new file is uploaded."""
    if uploaded_file:
        cur_hash = hash(uploaded_file.getvalue())
        if st.session_state.last_file_hash != cur_hash:
            st.session_state.answers = []
            st.session_state.last_file_hash = cur_hash
            if "_make_store" in st.session_state:
                del st.session_state["_make_store"]

_reset_answers_if_new_file()

# ---------------- Cached vector store ----------------
@st.cache_resource(show_spinner=False)
def _make_store(docs, api_key: str):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    return Chroma.from_documents(docs, embeddings)

# ---------------- QA chain builder ----------------
def build_qa_chain(file_bytes: bytes, api_key: str, query: str) -> str:
    raw_text = file_bytes.decode(errors="replace")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([raw_text])
    store = _make_store(docs, api_key)
    retriever = store.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key),
        retriever=retriever,
        chain_type="stuff",
    )
    return qa.run(query)

# ---------------- QA form ----------------
if uploaded_file:
    with st.form(key="qa_form"):
        query_text = st.text_input(
            "Enter your question:",
            placeholder="e.g. Summarise the main points."
        )
        submit_button = st.form_submit_button("Run QA")

    if submit_button:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            st.error("API key not set in Streamlit secrets.")
        elif not query_text:
            st.info("Please enter a question to ask.")
        else:
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
    st.info("Please upload a .txt file first.")
