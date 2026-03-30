import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")

CHAT_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 3500


def initialize_session_state() -> None:
    defaults = {
        "vector_store": None,
        "vectorizer": None,
        "chunk_docs": [],
        "chat_history": [],
        "indexed_files": [],
        "document_count": 0,
        "chunk_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_uploaded_pdfs(uploaded_files) -> List[Path]:
    saved_paths: List[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="pdf_rag_"))

    for uploaded_file in uploaded_files:
        target_path = temp_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)

    return saved_paths


def load_documents(pdf_paths: List[Path]) -> List[Document]:
    documents: List[Document] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        documents.extend(docs)

    return documents


def build_vector_store(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[TfidfVectorizer, Any, List[Document], int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunk_texts = [clean_text(chunk.page_content) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunk_texts)

    return vectorizer, matrix, chunks, len(chunks)


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Configuration")
        st.info(
            "This app uses free public Hugging Face models. The first run may take a minute while models download."
        )
        chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1200, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=200, step=25)
        top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4, step=1)
        st.caption("Answers use stable generation settings to reduce repetition on small local models.")

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
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "uploaded_files": uploaded_files,
        "process_clicked": process_clicked,
        "clear_clicked": clear_clicked,
    }


def clear_knowledge_base() -> None:
    st.session_state.vector_store = None
    st.session_state.vectorizer = None
    st.session_state.chunk_docs = []
    st.session_state.chat_history = []
    st.session_state.indexed_files = []
    st.session_state.document_count = 0
    st.session_state.chunk_count = 0


def build_prompt(context: str, question: str) -> str:
    return f"""
Answer the question using only the context from the uploaded PDFs.
If the context does not contain the answer, reply exactly:
I could not find the answer in the uploaded PDFs.
Give a complete answer in 3 to 6 sentences when the context supports it.
Use short bullet points if that makes the answer clearer.
Keep the response factual and in plain text.
Do not repeat phrases.

Context:
{context}

Question:
{question}
"""


@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHAT_MODEL)
    return tokenizer, model


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_context(context_docs: List[Document]) -> str:
    parts: List[str] = []
    total_chars = 0

    for index, doc in enumerate(context_docs, start=1):
        file_name = doc.metadata.get("source_file", "Unknown file")
        page_number = doc.metadata.get("page")
        page_label = f"{page_number + 1}" if page_number is not None else "?"
        body = clean_text(doc.page_content)
        if not body:
            continue

        snippet = f"[Source {index}: {file_name}, page {page_label}] {body}"
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining]
        parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n".join(parts)


def normalize_answer(answer: str) -> str:
    answer = clean_text(answer)
    if not answer:
        return "I could not find the answer in the uploaded PDFs."

    words = answer.split()
    if len(words) > 180:
        answer = " ".join(words[:180]).strip()

    repetitive = len(set(words[:20])) <= max(3, len(words[:20]) // 4) if words else False
    if repetitive:
        return "I could not generate a reliable answer from the uploaded PDFs."

    return answer


def retrieve_context_docs(question: str, top_k: int) -> List[Document]:
    if st.session_state.vectorizer is None or st.session_state.vector_store is None:
        return []

    query = clean_text(question)
    query_vector = st.session_state.vectorizer.transform([query])
    scores = cosine_similarity(query_vector, st.session_state.vector_store).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    docs: List[Document] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        docs.append(st.session_state.chunk_docs[idx])
    return docs


def answer_question(question: str, top_k: int) -> dict:
    tokenizer, model = load_llm()
    context_docs = retrieve_context_docs(question, top_k)
    if not context_docs:
        return {
            "answer": "I could not find the answer in the uploaded PDFs.",
            "context": [],
        }
    context = build_context(context_docs)
    prompt = build_prompt(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    generation_kwargs = {
        "max_new_tokens": 320,
        "do_sample": False,
        "num_beams": 4,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    output_ids = model.generate(**inputs, **generation_kwargs)
    answer = normalize_answer(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return {
        "answer": answer,
        "context": context_docs,
    }


def render_status() -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed files", len(st.session_state.indexed_files))
    col2.metric("Loaded pages", st.session_state.document_count)
    col3.metric("Text chunks", st.session_state.chunk_count)

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


def format_sources(context_docs: List[Document]) -> List[str]:
    seen = set()
    sources = []

    for doc in context_docs:
        file_name = doc.metadata.get("source_file", "Unknown file")
        page_number = doc.metadata.get("page")
        source_line = f"- `{file_name}`"
        if page_number is not None:
            source_line += f" (page {page_number + 1})"

        if source_line not in seen:
            seen.add(source_line)
            sources.append(source_line)

    return sources


def main() -> None:
    initialize_session_state()
    st.title("Bulk PDF RAG Assistant")
    st.write(
        "Upload multiple PDFs, build a searchable knowledge base, and ask questions grounded in those documents."
    )

    config = render_sidebar()

    if config["clear_clicked"]:
        clear_knowledge_base()
        st.success("Knowledge base cleared.")

    if config["process_clicked"]:
        if not config["uploaded_files"]:
            st.error("Upload at least one PDF file.")
        else:
            with st.spinner("Reading PDFs and building the vector index..."):
                pdf_paths = save_uploaded_pdfs(config["uploaded_files"])
                documents = load_documents(pdf_paths)
                vectorizer, vector_store, chunk_docs, chunk_count = build_vector_store(
                    documents=documents,
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"],
                )

            st.session_state.vectorizer = vectorizer
            st.session_state.vector_store = vector_store
            st.session_state.chunk_docs = chunk_docs
            st.session_state.chat_history = []
            st.session_state.indexed_files = [pdf.name for pdf in config["uploaded_files"]]
            st.session_state.document_count = len(documents)
            st.session_state.chunk_count = chunk_count
            st.success("PDFs processed successfully. You can start asking questions now.")

    render_status()
    render_chat_history()

    question = st.chat_input("Ask a question about the uploaded PDFs")
    if question:
        if st.session_state.vector_store is None:
            st.error("Process some PDF files first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the documents and drafting an answer..."):
                result = answer_question(
                    question=question,
                    top_k=config["top_k"],
                )
                answer = result["answer"]
                sources = format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources used", expanded=True):
                    for source in sources:
                        st.markdown(source)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
