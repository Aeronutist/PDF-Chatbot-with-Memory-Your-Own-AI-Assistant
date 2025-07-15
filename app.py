import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import pdfplumber
import os

st.set_page_config(page_title="Ask Your PDF – Vector QA", layout="centered")
st.title("📄 Ask Your PDF – Smart Search AI")
st.markdown("Upload a PDF and ask anything. This AI will search your document and answer using semantic retrieval.")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa

embedder, qa_model = load_models()

# PDF Upload
uploaded_file = st.file_uploader("📁 Upload PDF", type=["pdf"])
if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        raw_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

    # Chunking
    def chunk_text(text, size=500, overlap=50):
        chunks, start = [], 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    chunks = chunk_text(raw_text)
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.success(f"✅ PDF processed with {len(chunks)} chunks.")

    # Ask question
    question = st.text_input("💬 Ask a question from this PDF:")
    if question:
        q_embedding = embedder.encode([question])
        D, I = index.search(np.array(q_embedding), k=3)
        top_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(top_chunks)

        with st.spinner("🤖 Thinking..."):
            try:
                result = qa_model(question=question, context=context)
                st.markdown(f"### 🧠 Answer: `{result['answer']}`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                