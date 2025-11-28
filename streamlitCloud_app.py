import streamlit as st
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🦜🔗 Ask the Doc App", layout="centered")

# Load API key once at the top
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("API key not set in Streamlit secrets.")

# Session state
if "answers" not in st.session_state:
    st.session_state.answers = []

st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader(
    "Upload a plain-text article", type=["txt"]
)

def _reset_answers_if_new_file():
    if uploaded_file:
        cur_hash = hash(uploaded_file.getvalue())
        if st.session_state.get("last_file_hash") != cur_hash:
            st.session_state.answers = []
            st.session_state.last_file_hash = cur_hash
            if "_make_store" in st.session_state:
                del st.session_state["_make_store"]

_reset_answers_if_new_file()

# QA form
if uploaded_file and api_key:
    with st.form(key="qa_form"):
        query_text = st.text_input(
            "Enter your question:",
            placeholder="e.g. Summarise the main points."
        )
        submit_button = st.form_submit_button("Run QA")

    if submit_button and query_text:
        with st.spinner("Generating answer..."):
            try:
                # Build QA chain
                raw_text = uploaded_file.getvalue().decode(errors="replace")
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.create_documents([raw_text])
                embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
                store = Chroma.from_documents(docs, embeddings)
                retriever = store.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=api_key),
                    retriever=retriever,
                    chain_type="stuff",
                )
                answer = qa.run(query_text)
                st.subheader("Answer")
                st.write(answer)
                st.session_state.answers.append(answer)
            except Exception as e:
                st.error(f"Error running QA chain: {e}")
else:
    if not uploaded_file:
        st.info("Please upload a .txt file first.")
