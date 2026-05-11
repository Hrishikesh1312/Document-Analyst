# Document Analyst

Document Analyst is a privacy-first Streamlit desktop-style app for semantic search and local question-answering over PDFs, Markdown, and text files.

## Features

- Local document ingestion for `.pdf`, `.md`, `.markdown`, and `.txt`
- Semantic chunking with sentence-level similarity boundaries
- Persistent local vector store with `ChromaDB`
- Multi-turn chat over retrieved document context
- Inline source citations plus expandable source cards
- First-launch model download into `models/`
- Light/dark appearance toggle
- CPU-friendly local inference through `llama-cpp-python`

## Stack

- `Streamlit`
- `ChromaDB`
- `PyMuPDF`
- `sentence-transformers`
- `llama-cpp-python`
- `huggingface_hub`

## Run

If the local environment already exists:

```bash
.venv/bin/streamlit run app.py
```

If you need to install dependencies in a fresh environment:

```bash
python3 -m pip install -r requirements.txt
```

## First launch

1. Open the `Models & Settings` tab.
2. Click `Download Recommended Models`.
3. Wait for the embedding model and a GGUF file to download into `models/`.
4. Go to `Manage Documents`, choose a local folder, and build the index.
5. Use the `Chat` tab to ask grounded questions.

## Notes

- Chat history is session-only for privacy.
- The app assumes consumer hardware with CPU inference.
- If the default GGUF repo changes upstream, you can override it in `Models & Settings`.
