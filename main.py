import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_real_token_here"

# 1. Load PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(f"Number of pages loaded: {len(pages)}")

# 2. Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(pages)
print(f"Number of chunks created: {len(texts)}")

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create vector store
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
print("✅ Vectorstore created successfully.")

# 5. Hugging Face model
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=pipe)

# 6. Retrieval Q&A
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())

# 7. Chat loop
print("\n🗨️ PDF Chatbot Ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break
    answer = qa.invoke(query)["result"]  # Only show the answer
    print("Bot:", answer)
