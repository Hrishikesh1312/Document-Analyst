# Document Analyst

Document Analyst is a local retrieval-augmented generation application for querying document collections. It uses hybrid semantic and BM25 retrieval, reranking, persistent vector storage, and local GGUF inference. Documents, embeddings, conversations, and model files remain on the host system.

## Features

- Local ingestion of PDF, DOCX, PPTX, Markdown, and plain-text files
- Optional Tesseract OCR for scanned PDF pages
- Hybrid semantic and BM25 retrieval with relevance filtering and document diversification
- Local response generation through `llama.cpp` and GGUF models
- Inline citations and highlighted supporting passages
- Direct PDF links to cited pages
- Source filtering, pinning, and exclusion for subsequent queries
- Incremental indexing using SHA-256 file hashes
- Changed-file, duplicate-document, removal, and failure detection
- Background indexing with progress reporting, cancellation, and failed-file retry
- Named conversations with persistent local history
- Conversation export to Markdown and PDF
- Retrieval diagnostics with candidate scores and timing information

## Supported formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Supports optional OCR fallback |
| Microsoft Word | `.docx` | Open XML format only |
| Microsoft PowerPoint | `.pptx` | Open XML format only |
| Markdown | `.md`, `.markdown` | Parsed as text |
| Plain text | `.txt` | UTF-8 text |

Legacy `.ppt` files are not supported. Convert them to `.pptx` before indexing.

## Requirements

- Python 3.10 through 3.13
- A supported local GGUF model for response generation
- A sentence-transformer model for embeddings
- Tesseract system executable if OCR is enabled

Model files can be selected and downloaded from the application.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd Document-Analyst
python3 -m venv .venv
```

Activate the environment.

macOS and Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the application:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the application

Start the Streamlit interface:

```bash
python -m streamlit run app.py
```

The installed command provides the same entry point:

```bash
document-analyst
```

## Initial configuration

1. Open **Settings**.
2. Select and download an embedding model and a GGUF language model.
3. Configure retrieval and OCR settings if required.
4. Open **Library** and specify a local document directory.
5. Build the index.
6. Open **Ask** and query the indexed collection.

Model downloads may require several gigabytes of disk space, depending on the selected GGUF file.

## OCR configuration

OCR is disabled by default and requires the Tesseract executable.

macOS:

```bash
brew install tesseract
```

Ubuntu and Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

On Windows, install a Tesseract distribution and add its installation directory to `PATH`, or configure the full executable path under **Settings**.

## Data storage

Application state is stored outside the repository in the operating system's user-data directory.

| Platform | Default location |
|---|---|
| macOS | `~/Library/Application Support/Document Analyst` |
| Windows | `%LOCALAPPDATA%\Document Analyst` |
| Linux | `$XDG_DATA_HOME/document-analyst` or `~/.local/share/document-analyst` |

Set `DOCUMENT_ANALYST_HOME` before startup to override the default location:

```bash
export DOCUMENT_ANALYST_HOME=/path/to/application-data
python -m streamlit run app.py
```

The application-data directory contains settings, downloaded models, the Chroma database, the indexing manifest, and conversation history. Generated exports are returned directly by the application and do not need to be stored in the repository.

## Architecture

```text
Documents
    -> format-specific extraction and optional OCR
    -> semantic chunking
    -> sentence-transformer embeddings
    -> Chroma vector storage

Question
    -> semantic search and BM25 search
    -> score fusion, reranking, and diversification
    -> retrieved source context
    -> local GGUF inference through llama.cpp
    -> cited response
```

Primary components:

- Streamlit: application interface and session management
- ChromaDB: persistent vector storage
- Sentence Transformers: document and query embeddings
- `llama-cpp-python`: local GGUF inference
- PyMuPDF: PDF extraction and page rendering
- `python-docx`: DOCX extraction
- `python-pptx`: PPTX extraction
- ReportLab: PDF conversation export

## Development

Install development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the test suite:

```bash
python -m pytest
```

Run the compilation check:

```bash
python -m compileall -q src app.py
```

Native dependencies such as PyTorch and `llama.cpp` may require platform-specific wheels. Use a platform-specific lock file for reproducible deployments.
