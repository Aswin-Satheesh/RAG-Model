# PDF RAG Assistant

This project now includes a Streamlit-based RAG application built with LangChain so you can:

- Upload multiple PDF files in one batch
- Convert the PDFs into searchable chunks
- Ask questions grounded in the uploaded documents
- See which files and pages were used for each answer

## Setup

1. Create or activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

The app uses a free public Hugging Face model for answering questions, so no OpenAI key is required. On the first run it will download the chat model locally.

## How it works

- PDFs are uploaded in bulk through the Streamlit sidebar.
- LangChain loads and splits the PDF text into chunks.
- PDF chunks are indexed locally with TF-IDF for similarity search.
- The most relevant chunks are sent to a Hugging Face text generation model.

## Notes

- The app uses local TF-IDF retrieval and `google/flan-t5-base` for answers.
- `requirements.txt` includes `streamlit>=1.33,<2.0`.
- If the answer is not present in the uploaded PDFs, the assistant is prompted to say so.
- Reprocessing files replaces the current in-memory knowledge base for the session.
