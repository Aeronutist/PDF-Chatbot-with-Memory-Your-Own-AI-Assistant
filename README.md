# 🤖 Ask Your PDF – Vector QA App

A smart AI app that reads your PDF, splits it into chunks, and answers your natural language questions using semantic search and transformers.  
Built with FAISS + Sentence Transformers + Hugging Face QA models.

🔗 **Live App**: [Click to Open]( https://pdf-chatbot-with-memory-your-own-ai-assistant-nmsjldiszsgoyrmb.streamlit.app/ ) 
📦 **Supports**: PDF  
💬 **Search Type**: Semantic / Vector-Based  
🧠 **Model**: Roberta (SQuAD2) + MiniLM-L6

---

## 🚀 Features

- 📁 Upload any `.pdf` document
- ✂️ Text is smartly chunked with overlap
- 🧠 Embeddings created using `sentence-transformers`
- 🔍 Fast semantic retrieval using `faiss`
- 🤖 QA powered by `deepset/roberta-base-squad2`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `streamlit` | Web interface |
| `pdfplumber` | Extract text from PDFs |
| `sentence-transformers` | Convert chunks to semantic vectors |
| `faiss-cpu` | Vector DB for fast search |
| `transformers` | QA model (Roberta) |

---

## 📄 Example Questions

Try uploading:
- A research article → Ask "What is the conclusion?"
- A legal PDF → Ask "What is the penalty for violation?"
- A user manual → Ask "How to reset the device?"

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
