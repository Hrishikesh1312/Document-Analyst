# Document Analyst

Document Analyst is a privacy-first local RAG application for PDF, DOCX, PPTX,
Markdown, and text files. It runs on macOS, Windows, and Linux and keeps documents,
embeddings, vector data, and inference on the local machine.

Retrieval combines semantic vector search with local BM25 keyword search, then
reranks and diversifies the candidate pool before generation. An optional
diagnostics panel exposes candidate scores, selected evidence, document
coverage, filters, and retrieval timings.

Indexing is incremental and records SHA-256 file hashes, timestamps, per-file
status, duplicate relationships, failures, and removals. Long jobs run in the
background with safe cancellation and failed-file retry controls.

Chats are stored locally as named conversations and can be renamed, deleted,
or exported with citations to Markdown and PDF. Source cards highlight the
best-matching passage, link to PDF pages, support evidence search, and provide
one-response pin/exclude controls.

## What is portable now

- Runtime data uses the native per-user application-data location instead of
  writing into the source checkout.
- Paths use `pathlib`; document traversal does not follow symbolic links.
- Settings are validated and replaced atomically, with corrupt files backed up.
- Native ML runtimes load only when indexing or answering requires them.
- Chroma handles are cached across Streamlit reruns and large writes are batched.
- Re-indexing removes stale chunks, and rendered document content is HTML-escaped.

Default data locations:

- macOS: `~/Library/Application Support/Document Analyst`
- Windows: `%LOCALAPPDATA%\Document Analyst`
- Linux: `$XDG_DATA_HOME/document-analyst` or `~/.local/share/document-analyst`

Set `DOCUMENT_ANALYST_HOME` before launch to use a custom location.

## Install

Python 3.10–3.13 is supported. A fresh virtual environment is strongly
recommended.

### macOS

Install Python from python.org or Homebrew. On Apple Silicon, make sure Python
and the terminal use the same architecture (`arm64`). Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m streamlit run app.py
```

Optional OCR requires the system Tesseract executable:

```bash
brew install tesseract
```

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m streamlit run app.py
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m streamlit run app.py
```

After installation, `document-analyst` is also available as a launcher.

## First launch

1. Open **Models & Settings** and download the selected models.
2. Open **Manage Documents**, choose a readable folder, and build the index.
3. Return to **Chat** and ask questions about the indexed content.

The first download can be several gigabytes. Chat history remains session-only.

## Verify a development checkout

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m compileall -q src app.py
```

For reproducible deployments, create a lock file on each target architecture;
PyTorch and llama.cpp wheels differ between operating systems and CPU types.
