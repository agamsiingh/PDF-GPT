import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# -----------------------
# Hugging Face API Token
# -----------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_real_token_here"

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("📖 PDF  with Multiple PDFs")
st.write("Upload one or more PDFs and chat with them like ChatGPT 🤖")

# -----------------------
# Sidebar File Uploader
# -----------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------
# Session State
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "qa" not in st.session_state:
    st.session_state.qa = None

# -----------------------
# Answer Length Selector
# -----------------------
length_option = st.sidebar.radio(
    "Select Answer Length",
    ["Short", "Medium", "Long"],
    index=2
)

length_map = {
    "Short": "Answer briefly in 2-3 sentences.",
    "Medium": "Answer in detail with at least one paragraph (5-6 sentences).",
    "Long": "Answer thoroughly in multiple paragraphs with examples and insights."
}

# -----------------------
# Process PDFs
# -----------------------
if uploaded_files:
    all_pages = []

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()
        all_pages.extend(pages)

    st.success(f"✅ Loaded {len(uploaded_files)} PDF(s) with {len(all_pages)} total pages.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(all_pages)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

    # Hugging Face LLM (use XXL for longer answers, Large if RAM is low)
    pipe = pipeline("text2text-generation", model="google/flan-t5-xxl", max_length=1500)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Custom Prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a knowledgeable assistant. Use the context below to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            f"{length_map[length_option]}\n\n"
            "Answer:"
        )
    )

    # Retrieval QA
    st.session_state.qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template}
    )

# -----------------------
# Chat Interface
# -----------------------
if st.session_state.qa:
    user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        answer = st.session_state.qa.invoke(user_input)["result"]

        # Save chat history
        st.session_state.history.append({"question": user_input, "answer": answer})

    # Display chat history
    for chat in st.session_state.history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Clear Chat
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.session_state.qa = None
        st.rerun()
