# 📖 PDF Chatbot

A **Streamlit-based chatbot** that allows you to upload one or more PDF files and ask questions about their content. Powered by **Hugging Face LLMs** for fast, cloud-based answers.  

---

## 🔹 Features
- Upload multiple PDFs at once.
- Ask questions about PDF content like ChatGPT.
- Adjustable answer length: Short, Medium, or Long.
- Uses **Hugging Face API** for fast LLM responses, so your local CPU is not overloaded.
- Stores conversation history within the session.
- Built with **Streamlit**, **LangChain**, and **Chroma** for embeddings.

---

## 🔹 Demo
<img width="959" height="476" alt="Image" src="https://github.com/user-attachments/assets/7f557769-e498-4122-9b21-62dbbf8ca99e" /> 
<br>
<img width="959" height="448" alt="Image" src="https://github.com/user-attachments/assets/26b0e3b2-db97-49c1-b090-63e38be04e2a" />
<br>
<img width="959" height="475" alt="Image" src="https://github.com/user-attachments/assets/4465cb2b-c6c4-43f4-8250-88189bd8c253" />
---

## 🔹 How It Works
1. Upload PDFs via the sidebar.
2. PDFs are split into chunks using `RecursiveCharacterTextSplitter`.
3. Text chunks are converted into embeddings using **Hugging Face sentence-transformers**.
4. Embeddings are stored in a **Chroma vector database** for fast retrieval.
5. Questions are sent to a **Hugging Face LLM** (e.g., `flan-t5-large`) via LangChain.
6. Answers are generated using the context retrieved from your PDFs.

---

## 🔹 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
(Optional) Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set your Hugging Face API token:

bash
Copy code
export HUGGINGFACEHUB_API_TOKEN="hf_your_real_token_here"  # Linux/Mac
set HUGGINGFACEHUB_API_TOKEN=hf_your_real_token_here       # Windows
Run the app:

bash
Copy code
streamlit run app.py
Open your browser at http://localhost:8501.

🔹 Usage
Upload one or more PDF files using the sidebar.

Select the desired answer length:

Short: 2-3 sentences.

Medium: 1 paragraph (5-6 sentences).

Long: Multiple paragraphs with examples and insights.

Type your question in the chat input.

View answers and conversation history in the main window.

Click Clear Chat to reset the session.

🔹 Requirements
Python 3.10+

Streamlit

LangChain

Transformers

Chroma

PyPDFLoader

requirements.txt example:

nginx
Copy code
streamlit
langchain
langchain-huggingface
transformers
chromadb
🔹 Optional Deployment
Streamlit Cloud: Free, fast, easy. No local CPU usage.

Docker: Run on any cloud server with containerization.

🔹 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

🔹 License
MIT License.

Made with ❤️ using Streamlit, LangChain & Hugging Face.
