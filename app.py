import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(page_title="Local PDF RAG", layout="wide")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
PRIMARY_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "mistral"
AVAILABLE_OLLAMA_MODELS = ["mistral", "llama3"]
TOP_K_RETRIEVAL = 8
FINAL_CONTEXT_DOCS = 4
MAX_CONTEXT_CHARS = 3600
EMBED_BATCH_SIZE = 32


def initialize_session_state() -> None:
    defaults = {
        "index": None,
        "chunks": [],
        "chat_history": [],
        "indexed_files": [],
        "document_count": 0,
        "chunk_count": 0,
        "embedding_model_name": None,
        "knowledge_base_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z])\s(?=[A-Za-z]\s){4,}", "", text)
    return text.strip()


def save_uploaded_pdfs(uploaded_files) -> List[Path]:
    saved_paths: List[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="local_rag_"))

    for uploaded_file in uploaded_files:
        target_path = temp_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)

    return saved_paths


def load_documents(pdf_paths: List[Path]) -> Tuple[List[Document], List[str]]:
    documents: List[Document] = []
    warnings: List[str] = []

    for pdf_path in pdf_paths:
        try:
            docs = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            warnings.append(f"Could not read `{pdf_path.name}`: {exc}")
            continue

        valid_pages = 0
        for doc in docs:
            cleaned = clean_text(doc.page_content)
            if not cleaned:
                continue
            doc.page_content = cleaned
            doc.metadata["source_file"] = pdf_path.name
            documents.append(doc)
            valid_pages += 1

        if valid_pages == 0:
            warnings.append(f"`{pdf_path.name}` did not contain extractable text.")

    return documents, warnings


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    cleaned_chunks: List[Document] = []

    for index, chunk in enumerate(chunks):
        content = clean_text(chunk.page_content)
        if len(content) < 40:
            continue
        chunk.page_content = content
        chunk.metadata["chunk_id"] = index
        cleaned_chunks.append(chunk)

    return cleaned_chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> Tuple[str, SentenceTransformer]:
    for model_name in (PRIMARY_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL):
        try:
            model = SentenceTransformer(model_name)
            return model_name, model
        except Exception:
            continue
    raise RuntimeError(
        "Unable to load an embedding model. Install sentence-transformers and ensure Hugging Face downloads work."
    )


def create_embeddings(chunks: List[Document], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    if not chunks:
        raise ValueError("No chunks available to embed.")

    _, model = load_embedding_model()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.size == 0:
        raise ValueError("Embeddings are empty; cannot build FAISS index.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def lexical_overlap_score(query: str, text: str) -> float:
    query_terms = {token for token in re.findall(r"\w+", query.lower()) if len(token) > 2}
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"\w+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def rerank_documents(
    query: str,
    candidates: List[Tuple[Document, float]],
    final_k: int = FINAL_CONTEXT_DOCS,
) -> List[Document]:
    rescored: List[Tuple[float, Document]] = []

    for doc, semantic_score in candidates:
        lexical_score = lexical_overlap_score(query, doc.page_content)
        score = (semantic_score * 0.75) + (lexical_score * 0.25)
        rescored.append((score, doc))

    rescored.sort(key=lambda item: item[0], reverse=True)

    selected: List[Document] = []
    seen_keys = set()
    for _, doc in rescored:
        key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", -1),
            doc.page_content[:160],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(doc)
        if len(selected) >= final_k:
            break

    return selected


def retrieve_documents(
    question: str,
    top_k: int = TOP_K_RETRIEVAL,
    final_k: int = FINAL_CONTEXT_DOCS,
) -> List[Document]:
    if st.session_state.index is None or not st.session_state.chunks:
        return []

    _, model = load_embedding_model()
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    limit = min(top_k, len(st.session_state.chunks))
    scores, indices = st.session_state.index.search(query_embedding, limit)

    candidates: List[Tuple[Document, float]] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        candidates.append((st.session_state.chunks[index_position], float(score)))

    return rerank_documents(question, candidates, final_k=final_k)


def build_context(documents: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total_chars = 0
    seen_bodies = set()

    for doc in documents:
        body = clean_text(doc.page_content)
        if not body or body in seen_bodies:
            continue
        seen_bodies.add(body)

        file_name = doc.metadata.get("source_file", "Unknown file")
        page_number = doc.metadata.get("page")
        page_label = page_number + 1 if page_number is not None else "?"
        section = f"[File: {file_name} | Page: {page_label}]\n{body}"

        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip()

        parts.append(section)
        total_chars += len(section) + 2

    return "\n\n".join(parts)


def build_prompt(context: str, question: str) -> str:
    return f"""You are a retrieval-augmented assistant.
Answer ONLY from the supplied document context.
If the answer is not in the context, say exactly:
I could not find the answer in the documents.
Do not hallucinate.
Keep the answer concise in 3 to 6 sentences.

Context:
{context}

Question:
{question}

Answer:
"""


def query_ollama(prompt: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 220,
            "repeat_penalty": 1.15,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to reach Ollama. Make sure Ollama is running locally and the selected model is pulled."
        ) from exc

    if response.status_code >= 400:
        try:
            details = response.json().get("error", response.text)
        except ValueError:
            details = response.text
        raise RuntimeError(f"Ollama error: {details}")

    data = response.json()
    answer = clean_text(data.get("response", ""))
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


def normalize_answer(answer: str) -> str:
    answer = clean_text(answer)
    if not answer:
        return "I could not find the answer in the documents."

    if re.search(r"(.{1,40})\1{2,}", answer):
        return "I could not find the answer in the documents."

    words = answer.split()
    if len(words) > 150:
        answer = " ".join(words[:150]).strip()

    return answer


def generate_answer(question: str, model_name: str, top_k: int) -> Dict[str, Any]:
    retrieved_docs = retrieve_documents(question, top_k=top_k, final_k=min(5, top_k))
    if not retrieved_docs:
        return {
            "answer": "I could not find the answer in the documents.",
            "sources": [],
        }

    context = build_context(retrieved_docs)
    if not context:
        return {
            "answer": "I could not find the answer in the documents.",
            "sources": [],
        }

    prompt = build_prompt(context=context, question=question)
    answer = normalize_answer(query_ollama(prompt, model_name))
    return {
        "answer": answer,
        "sources": format_sources(retrieved_docs),
    }


def format_sources(context_docs: List[Document]) -> List[str]:
    sources: List[str] = []
    seen = set()

    for doc in context_docs:
        file_name = doc.metadata.get("source_file", "Unknown file")
        page_number = doc.metadata.get("page")
        source = f"- `{file_name}`"
        if page_number is not None:
            source += f" (page {page_number + 1})"
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources


def knowledge_base_signature(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{chunk_size}:{chunk_overlap}".encode("utf-8"))

    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getbuffer())

    return digest.hexdigest()


def clear_knowledge_base() -> None:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.chat_history = []
    st.session_state.indexed_files = []
    st.session_state.document_count = 0
    st.session_state.chunk_count = 0
    st.session_state.knowledge_base_id = None


@st.cache_data(show_spinner=False, ttl=10)
def get_ollama_models() -> List[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        model_names = [model.get("name", "") for model in data.get("models", []) if model.get("name")]
        if model_names:
            return model_names
    except requests.RequestException:
        pass
    return AVAILABLE_OLLAMA_MODELS


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Configuration")
        ollama_models = get_ollama_models()
        default_index = 0
        for index, model_name in enumerate(ollama_models):
            if model_name == DEFAULT_OLLAMA_MODEL or model_name.startswith(f"{DEFAULT_OLLAMA_MODEL}:"):
                default_index = index
                break
        ollama_model = st.selectbox(
            "Ollama model",
            ollama_models,
            index=default_index,
        )
        chunk_size = st.slider("Chunk size", min_value=800, max_value=1200, value=1000, step=50)
        chunk_overlap = st.slider("Chunk overlap", min_value=150, max_value=200, value=180, step=10)
        top_k = st.slider("Top-K retrieval", min_value=3, max_value=8, value=TOP_K_RETRIEVAL, step=1)

        st.caption("Embeddings use `BAAI/bge-base-en-v1.5` with fallback to `all-MiniLM-L6-v2`.")
        st.caption("Answers are generated locally through Ollama at `http://localhost:11434`.")
        if ollama_models == AVAILABLE_OLLAMA_MODELS:
            st.caption("If your installed Ollama models do not appear here, confirm `ollama serve` is running.")

        st.divider()
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process_clicked = st.button("Process PDFs", type="primary", use_container_width=True)
        clear_clicked = st.button("Clear Knowledge Base", use_container_width=True)

    return {
        "ollama_model": ollama_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "uploaded_files": uploaded_files,
        "process_clicked": process_clicked,
        "clear_clicked": clear_clicked,
    }


def render_status() -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed files", len(st.session_state.indexed_files))
    col2.metric("Loaded pages", st.session_state.document_count)
    col3.metric("Text chunks", st.session_state.chunk_count)

    if st.session_state.embedding_model_name:
        st.caption(f"Embedding model: `{st.session_state.embedding_model_name}`")
    if st.session_state.indexed_files:
        st.caption("Indexed PDFs: " + ", ".join(st.session_state.indexed_files))


def render_chat_history() -> None:
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources used"):
                    for source in item["sources"]:
                        st.markdown(source)


def build_knowledge_base(config: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    uploaded_files = config["uploaded_files"]
    if not uploaded_files:
        raise ValueError("Upload at least one PDF file.")

    current_signature = knowledge_base_signature(
        uploaded_files,
        config["chunk_size"],
        config["chunk_overlap"],
    )

    if st.session_state.knowledge_base_id == current_signature and st.session_state.index is not None:
        return warnings

    pdf_paths = save_uploaded_pdfs(uploaded_files)
    documents, load_warnings = load_documents(pdf_paths)
    warnings.extend(load_warnings)

    if not documents:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    chunks = chunk_documents(
        documents=documents,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    if not chunks:
        raise ValueError("The uploaded PDFs did not produce any valid chunks.")

    embedding_model_name, _ = load_embedding_model()
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.chat_history = []
    st.session_state.indexed_files = [pdf.name for pdf in uploaded_files]
    st.session_state.document_count = len(documents)
    st.session_state.chunk_count = len(chunks)
    st.session_state.embedding_model_name = embedding_model_name
    st.session_state.knowledge_base_id = current_signature

    return warnings


def main() -> None:
    initialize_session_state()
    st.title("Production-Ready Local PDF RAG")
    st.write(
        "Upload PDFs, build a local FAISS index with sentence-transformer embeddings, and ask grounded questions with Ollama."
    )

    config = render_sidebar()

    if config["clear_clicked"]:
        clear_knowledge_base()
        st.success("Knowledge base cleared.")

    if config["process_clicked"]:
        try:
            with st.spinner("Loading PDFs, chunking text, creating embeddings, and building the FAISS index..."):
                warnings = build_knowledge_base(config)
            st.success("PDFs processed successfully. You can start asking questions.")
            for warning in warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(str(exc))

    render_status()
    render_chat_history()

    question = st.chat_input("Ask a question about the uploaded PDFs")
    if question:
        if st.session_state.index is None:
            st.error("Process PDF files first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving context and querying Ollama..."):
                    result = generate_answer(
                        question=question,
                        model_name=config["ollama_model"],
                        top_k=config["top_k"],
                    )
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources used", expanded=True):
                        for source in result["sources"]:
                            st.markdown(source)
            except Exception as exc:
                result = {"answer": f"Error: {exc}", "sources": []}
                st.error(result["answer"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )


if __name__ == "__main__":
    main()
