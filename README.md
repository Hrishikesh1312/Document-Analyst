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

1. In the welcome dialog, choose **Proceed with recommended settings** to download the default models.
2. In **Library**, open **Add a document folder**, select your local directory, and click **Build Index**.
3. Open **Ask** to chat with your documents. Use **Search scope** to limit the documents searched.
4. Use **Conversations** to switch or export chats, and **Search tools** for evidence search and diagnostics.

Library shows compact document rows. **View passages** opens a paginated dialog; **Details**
shows the file path and document management actions. **Performance options** stays collapsed
until needed, and indexing progress remains visible outside the setup controls.

In **Settings**, model readiness and the download action appear first. Open **Change models**
to choose alternatives, or enable advanced settings for retrieval, chunking, and OCR controls.

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

### Faster indexing

In Settings, enable advanced settings and open Indexing. Choose **Fast — sentence boundaries**
under **Chunking method**, then save. This uses sentence boundaries, chunk size, and overlap
without running a model on each sentence to choose boundaries. Finished chunks still receive
embeddings for semantic search. **Semantic — topic-aware boundaries** remains the default.
Both modes embed passages in configurable batches (32 by default), with progress and cancellation between batches.
The speedup depends on the document and hardware; final chunk embedding still takes time.
To apply a changed method to previously indexed documents, rebuild using **Replace existing
index contents** in Library (this replaces the entire current index with the selected folder).

### Embedding GPU acceleration

Embedding models select a device automatically on first use: **CUDA/MPS → CPU**.
CUDA requires a CUDA-enabled PyTorch installation and a compatible GPU/driver. On supported
Macs, PyTorch uses Apple's MPS backend. CPU is selected when neither backend is available
to PyTorch. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for the build appropriate to your machine.

The selected device is shown in Settings after the model loads. The same cached model handles
semantic chunking, final chunk embeddings, and search-query embeddings with the configured batch size.
Embeddings are normalized and returned to host memory for storage and retrieval as before.
Model loading remains lazy, so opening Settings does not initialize PyTorch.
If an available GPU fails during loading or inference, the error is reported rather than
silently switching to CPU. This change applies to sentence-transformer embeddings; GGUF
answer generation still uses its existing llama.cpp configuration.

### Progress, performance controls, and library search

Indexing shows the current stage, file, page or passage counts, live elapsed time, and embedding
device. The overall bar advances within each file; it is an approximate workload measure rather
than a prediction of wall-clock time. After several embedding batches, the panel shows passages
per second and an estimated time remaining for that stage. Completion refreshes the document
list automatically and preserves total elapsed time.

Library's **Indexing performance** controls offer these presets:

| Preset | Chunking | Target chunk size | Overlap |
| --- | --- | --- | --- |
| Fast | Sentence boundaries | 1,000 characters | 100 characters |
| Balanced | Sentence boundaries | 700 characters | 100 characters |
| Quality | Semantic topic boundaries | 500 characters | 100 characters |

Choose a preset and embedding batch size (1–256), then click **Apply performance settings**.
Presets preserve model, retrieval, OCR, and document-folder settings. Existing documents need
a rebuild to apply chunking changes; batch-size changes affect subsequent embedding calls.
The optional **Benchmark batch sizes** compares 8, 16, 32, 64, and 128 on the same sample text,
excluding model warmup. It reports results without changing settings or indexing documents.
Results depend on text length and hardware. Reduce the batch size if device memory is exhausted.

Search Library by document name or folder path, combine status and file-type filters, and clear
filters to restore all results. Documents display ten per page. **View passages** retrieves ten
stored chunks per page for the selected document, in source order with page references, without
loading stored embeddings or rerunning the model.
