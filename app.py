# app.py
import os
import tempfile
import traceback
import streamlit as st

# Try to import the same packages you used originally.
# If your environment uses slightly different package names, adapt imports accordingly.
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import CharacterTextSplitter as RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.vectorstores import Chroma

# Embedding + pipeline wrappers (these imports may be from community packages depending on your env)
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
except Exception:
    # If langchain_huggingface is not available, import fallback names (may need to install)
    HuggingFaceEmbeddings = None
    HuggingFacePipeline = None

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# -----------------------
# Config / Page Setup
# -----------------------
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("📖 PDF Genius")
st.write("Upload PDF(s). The app will create embeddings and let you ask questions.")

# track temporary files so we can cleanup later
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []

if "history" not in st.session_state:
    st.session_state.history = []

if "qa" not in st.session_state:
    st.session_state.qa = None

# Simple model selection (smaller models are faster / less memory-hungry)
MODEL_NAME = st.sidebar.selectbox(
    "Choose local HF model (start with small if you have limited RAM)",
    ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xxl"],
    index=0,
)
EMBEDDING_MODEL = st.sidebar.text_input(
    "Embedding model (HuggingFace)", "sentence-transformers/all-MiniLM-L6-v2"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files (can select many)",
    type=["pdf"],
    accept_multiple_files=True,
)

# -----------------------
# Helper: safe write uploaded file to temp file
# -----------------------
def save_uploaded_file_to_temp(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    st.session_state.temp_files.append(tmp.name)
    return tmp.name

# -----------------------
# Process uploaded PDFs
# -----------------------
if uploaded_files:
    try:
        all_pages = []
        with st.spinner("Saving uploaded PDFs and loading pages..."):
            for uploaded in uploaded_files:
                tmp_path = save_uploaded_file_to_temp(uploaded)
                st.sidebar.text(f"Saved: {uploaded.name} -> {tmp_path}")
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()  # returns list of Document objects
                all_pages.extend(pages)

        st.success(f"Loaded {len(uploaded_files)} PDF(s) with {len(all_pages)} pages.")
    except Exception as e:
        st.error("Error while loading PDFs. See details below.")
        st.exception(e)
        st.stop()

    # Split into chunks
    try:
        with st.spinner("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(all_pages)  # list of Documents
        st.write(f"Created {len(docs)} chunks/documents.")
    except Exception as e:
        st.error("Error while splitting documents.")
        st.exception(e)
        st.stop()

    # Create embeddings + vector store
    try:
        with st.spinner("Creating embeddings and vector store (Chroma)..."):
            if HuggingFaceEmbeddings is None:
                raise RuntimeError("HuggingFaceEmbeddings import failed. Install 'langchain_huggingface' or adjust imports.")

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
            # try persist if supported
            try:
                vectordb.persist()
            except Exception:
                pass
        st.success("Vector store ready.")
    except Exception as e:
        st.error("Error while creating embeddings/vectorstore.")
        st.exception(e)
        st.stop()

    # Load a Transformers pipeline and wrap with LangChain LLM wrapper
    try:
        with st.spinner(f"Loading text2text-generation pipeline ({MODEL_NAME})... (this can take a while)"):
            # device=-1 ensures CPU. Change to device=0 if GPU available and torch detects it.
            pipe = pipeline("text2text-generation", model=MODEL_NAME, max_length=512, device=-1)
            if HuggingFacePipeline is None:
                # fallback: wrap manually minimal adapter if langchain_huggingface is not installed
                from langchain.llms import HuggingFaceHub  # fallback-ish; may not match exact usage
                # Note: if you don't have langchain_huggingface, prefer installing it or using HuggingFaceHub
                llm = None
                raise RuntimeError("HuggingFacePipeline wrapper not found. Install 'langchain_huggingface' or adjust code to use HuggingFaceHub.")
            else:
                llm = HuggingFacePipeline(pipeline=pipe)
        st.success("LLM pipeline loaded.")
    except Exception as e:
        st.error("Error while loading model pipeline.")
        st.exception(e)
        st.stop()

    # Prompt template (simple)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the context below to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer succinctly and clearly. If the answer is not in the context, say you don't know.\n\n"
            "Answer:"
        ),
    )

    # Create Retrieval QA chain
    try:
        with st.spinner("Creating RetrievalQA chain..."):
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt_template},
            )
            st.session_state.qa = qa
        st.success("Ready — ask questions about your PDFs below.")
    except Exception as e:
        st.error("Error while creating RetrievalQA chain.")
        st.exception(e)
        st.stop()

# -----------------------
# Chat interface
# -----------------------
if st.session_state.qa:
    st.markdown("### Ask a question about the uploaded PDFs")
    # Use a simple text_input + button so it works broadly
    user_input = st.text_input("Your question", key="user_question")
    if st.button("Ask"):
        if user_input:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    # use .run for a simple single string interface
                    answer = st.session_state.qa.run(user_input)
            except Exception as e:
                st.error("Error while generating the answer.")
                st.exception(e)
                answer = "Sorry — an error occurred. See the exception above."

            st.session_state.history.append({"question": user_input, "answer": answer})

    # display history
    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Assistant:** {chat['answer']}")

    # clear button + cleanup
    if st.button("Clear Chat and Cleanup"):
        st.session_state.history = []
        st.session_state.qa = None
        # remove temp files we created
        for p in list(st.session_state.temp_files):
            try:
                os.remove(p)
            except Exception:
                pass
        st.session_state.temp_files = []
        st.experimental_rerun()
else:
    st.info("Upload PDFs on the left to get started.")

