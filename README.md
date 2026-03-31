# 🚀 Local PDF RAG Assistant (Production-Ready)

This project is a fully local, production-style Retrieval-Augmented Generation (RAG) system built with Streamlit, FAISS, sentence-transformers, and Ollama.

It allows you to:

* 📂 Upload multiple PDF files
* 🔍 Convert them into semantic embeddings
* ⚡ Perform fast similarity search using FAISS
* 💬 Ask questions grounded strictly in your documents
* 📄 View sources (file name + page number) for every answer
* 🧠 Run everything locally — no paid APIs required

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Embeddings:** BAAI/bge-base-en-v1.5 (fallback: all-MiniLM-L6-v2)
* **Vector DB:** FAISS (IndexFlatIP)
* **LLM:** Ollama (Mistral / Llama3)
* **Document Loader:** PyPDFLoader
* **Text Splitting:** RecursiveCharacterTextSplitter

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install Ollama

Download and install from: https://ollama.com

Then pull a model:

```bash
ollama pull mistral
```

---

### 4. Start Ollama

```bash
ollama serve
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🧠 How it works

### 1. Document Processing

* PDFs are uploaded via Streamlit
* Text is cleaned and extracted
* Documents are split into chunks (800–1200 chars with overlap)

---

### 2. Embedding & Indexing

* Each chunk is converted into embeddings using BGE model
* Embeddings are normalized for cosine similarity
* Stored in FAISS for fast retrieval

---

### 3. Retrieval

* User question is converted into an embedding
* Top-K similar chunks are retrieved
* Hybrid reranking is applied:

  * Semantic similarity (75%)
  * Keyword overlap (25%)

---

### 4. Context Building

* Top chunks are combined into a context window (~3600 chars)
* Includes file name and page references
* Removes duplicates and noise

---

### 5. Answer Generation

* Context + question is sent to Ollama (local LLM)
* Model is instructed:

  * Answer only from context
  * Avoid hallucination
  * Return concise responses

---

## 🎛️ Configuration Options

* **Chunk Size:** Controls text splitting granularity
* **Chunk Overlap:** Maintains context continuity
* **Top-K Retrieval:** Number of chunks retrieved before reranking
* **Model Selection:** Choose between installed Ollama models

---

## ⚡ Features

* ✅ Fully local (no API cost)
* ✅ Semantic search (not keyword-based)
* ✅ Hybrid reranking for better accuracy
* ✅ Context-aware responses
* ✅ Source attribution (file + page)
* ✅ Caching for faster performance
* ✅ Error handling for empty or invalid inputs

---

## 📝 Notes

* Ollama must be running at: `http://localhost:11434`
* First run may take time due to model downloads
* Knowledge base is session-based (cleared on reset)
* Large PDFs may increase processing time
* Ensure enough RAM (recommended: 16GB for Mistral)

---

## 💥 Summary

This project demonstrates a **modern, production-style RAG pipeline** using:

* Semantic embeddings
* Vector search (FAISS)
* Hybrid reranking
* Local LLM inference

All running **completely offline and free**.

---
