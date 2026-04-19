# Using the Citation RAG Pipeline

This document explains how to set up the environment and run each step of the pipeline.

## 0. Environment Setup

The pipeline requires a Conda environment. If you encountered an error like `ModuleNotFoundError: No module named 'rank_bm25'`, you need to install the dependencies in your active environment (`rag_prod`).

```bash
# Activate your environment
conda activate rag_prod

# Install required Python packages
pip install rank-bm25 chromadb layoutparser pymupdf opencv-python requests numpy ollama
pip install langchain-ollama langchain-experimental

# Ensure you have 'pdftotext' installed on your system (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install poppler-utils
```

Also, ensure that your `ollama` server is running in the background and you have pulled the necessary models:
```bash
ollama run gemma4:latest
ollama pull nomic-embed-text
```

## 1. Extracting Citations (`agent1_extractor.py`)

This step reads a base PDF from `raw/p1.pdf` (make sure this file exists) and uses `pdftotext` to extract a list of citations.

**How to run:**
```bash
python agent1_extractor.py
```
**Output:** It generates an `extracted_citations.json` file containing a list of reference strings parsed from your PDF.

## 2. Fetching Papers (`agent2_fetcher.py`)

This agent reads the extracted citations, locates their DOI, and tries to download open-access PDF versions.

*Note: You may want to update the `email` variable inside `agent2_fetcher.py` before running to adhere to the Unpaywall API etiquette.*

**How to run:**
```bash
python agent2_fetcher.py
```
**Output:** 
- Creates a `pulled_pdfs/` directory and populates it with the downloaded PDFs.
- Generates `downloaded.json` (mapping local file paths back to the citation string).
- Generates `failed_downloads.json` (logging citations that couldn't be fetched and the reason why).

## 3. Ingesting Papers (`agent3_ingestor.py`)

This is the most compute-heavy step. It reads the downloaded PDS, extracts text and figures, uses `gemma4:latest` logic to describe figures, chunks the documents, and inserts everything into a ChromaDB Vector DB and a BM25 index.

*Ensure that your Detectron2 weights are present at `../model_final.pth` relative to the current directory.*

**How to run:**
```bash
python agent3_ingestor.py
```
**Output:**
- Creates a `../extracted_data/images` directory containing cropped screenshots of figures and tables.
- Creates/updates `../physics_vectordb` containing the Chroma vector embeddings.
- Creates `../bm25_index.pkl` containing the sparse index for lexical search.

## 4. Citation Assistant (`agent4_assistant.py`)

Once your data is fully ingested, you can query the assistant using the command line. Provide a snippet of your draft text using the `--text` argument.

**How to run:**
```bash
python agent4_assistant.py --text "Recent advances in multilayer graphene nanoribbons have demonstrated tunable impurity scattering properties, offering a pathway for highly specialized quantum electronic devices."
```

**Optional Arguments:**
- `--top_k`: Number of contexts retrieved from the vector database (default: 3). Example: `--top_k 5`.

**Output:**
The script performs a hybrid retrieval (sparse + dense) and asks `gemma4:latest` to automatically suggest how to insert a proper citation directly into your drafted text, providing an explanation of the retrieved document context.
