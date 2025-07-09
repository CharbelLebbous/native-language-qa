# 📚 Native Language QA on Documents

This prototype allows you to **ask questions in natural English** about the contents of any folder of documents (**PDF**, **DOCX**, **TXT**) and get **precise answers** with:
- The file name,
- The full path,
- The page number (for PDFs).

---

## 🚀 Features

✅ Loads and cleans multiple file types  
✅ Splits long text into manageable chunks  
✅ Creates vector embeddings for semantic search  
✅ Runs local question-answering using Ollama (`phi3:mini`)  
✅ Displays results in a simple **Streamlit** web interface

---

## 🛠️ How to Run

1️⃣ **Create a virtual environment**
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app/streamlit_app.py

4️⃣ Optional: run to check the test results based on our small dataset of questions, OR check the results in the test_results file
python app/test_pipline.py
```

---

## 📌 How Native Language Support Works

This uses the Ollama LLM (phi3:mini) locally.
All queries and answers are processed in English as the prototype only supports English at the moment.
To add multilingual support, you can:

- Plug in a multilingual embedding model
- Use a translation layer for queries and answers
- Fine-tune your prompt to handle other languages

---

## ⚙️ Model Architecture

Pipeline:

- Loader — Recursively loads PDFs, DOCX, TXT → cleans text → splits into overlapping chunks
- Embedding Indexer — Uses HuggingFaceEmbeddings (intfloat/e5-base-v2) → FAISS for fast similarity search
- QA — Uses StuffDocumentsChain with phi3:mini → Generates final answer with references
- Frontend — Streamlit app for input, answer, sources display

---

## ⚠️ Limitations

- Runs CPU only, slower for large document sets, potentiol laptop shutting down due to CPU temprature increase.
- Only supports English.
- No advanced OCR (Optical Character Recognition) for scanned PDFs.
- No persistent embedding store (cleared on restart).

---

## 🧩 Possible Improvements

- Add multilingual embeddings (e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Use GPU for faster embedding generation.
- Implement index saving/loading with FAISS persist.
- Add OCR for scanned PDFs (e.g., pytesseract).
- Add unit tests and automated eval metrics.
- Use LangSmith or LlamaIndex for production-grade RAG pipeline.
- Add feedback loops to improve the model over time.

---

## 📧 Contact
For any issues, reach out to: charbellebbousf@gmail.com.
