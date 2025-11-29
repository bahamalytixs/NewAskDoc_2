import sys
import importlib.util

if importlib.util.find_spec("chromadb") is None:
    st.error("chromadb is not installed!")
else:
    st.success("chromadb is installed correctly")

print("Python:", sys.version)
print("chromadb installed:", importlib.util.find_spec("chromadb") is not None)

import streamlit as st
#from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title='🦜🔗 Ask the Doc App', layout='centered')
st.title('🦜🔗 Ask the Doc App')

# ---------------- Session state for answers ----------------
if 'answers' not in st.session_state:
    st.session_state['answers'] = []

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader('Upload an article', type='txt')

# ---------------- Only show form if file uploaded ----------------
if uploaded_file:
    with st.form('qa_form'):
        # Query text input
        query_text = st.text_input(
            'Enter your question:',
            placeholder='Please provide a short summary.'
        )

        # API key is loaded from Streamlit secrets
        api_key = st.secrets.get('OPENAI_API_KEY', '')

        submitted = st.form_submit_button('Submit')

        if submitted:
            if not api_key:
                st.error("API key not found in Streamlit secrets!")
            elif not query_text:
                st.warning("Please enter a question.")
            else:
                with st.spinner('Calculating...'):
                    try:
                        # Read file and split into chunks
                        raw_text = uploaded_file.read().decode()
                        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        docs = splitter.create_documents([raw_text])

                        # Create embeddings and vectorstore
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        db = Chroma.from_documents(docs, embeddings)
                        retriever = db.as_retriever()

                        # Create QA chain with modern ChatOpenAI
                        llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
                        qa = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            chain_type="stuff"
                        )

                       # Get answer
                        answer = qa.run(query_text)
                        st.session_state['answers'].append(answer)
                        st.success("Answer generated successfully!")
                        st.info(answer)

                    except Exception as e:
                        st.error(f"Error running QA chain: {e}")

# ---------------- Show info if no file uploaded ----------------
else:
    st.info("Please upload a .txt file to start.")
