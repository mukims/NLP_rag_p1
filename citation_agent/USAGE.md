# Using the Citation RAG Pipeline

This document explains how to set up the environment and run the pipeline, either fully automated via the Orchestrator, or manually step-by-step.

## 0. Environment Setup

The pipeline requires a Conda environment. 

```bash
# Activate your environment
conda activate rag_prod

# Install required Python packages
pip install rank-bm25 chromadb layoutparser pymupdf opencv-python requests numpy ollama watchdog
pip install langchain-ollama langchain-experimental

# Ensure you have 'pdftotext' installed on your system (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install poppler-utils
```

Also, ensure that your `ollama` server is running in the background and you have pulled the necessary models:
```bash
ollama run gemma4:latest
ollama pull nomic-embed-text
```

Ensure your Detectron2 weights are present at `../model_final.pth` relative to the current directory.

---

## 🚀 Fully Automated Mode (Recommended)

The easiest way to use the system is to run the **Master Orchestrator**. It will automatically trigger the ingestion of PDFs and the citation of text drafts.

**How to run:**
```bash
python master_orchestrator.py
```

**Workflow:**
1. **Ingesting Papers**: Simply drop a new PDF into the `raw/` directory. The orchestrator will detect it, wait 30 seconds for you to drop any additional files, and then automatically run Agents 1, 2, and 3 (in parallel).
2. **Citing Drafts**: Save a text file (e.g., `my_paper.txt`) into the `drafts/` directory. The orchestrator will instantly detect the save, run Agent 5, and output a `my_paper_cited.txt` file along with your bibliography JSON right next to it!

---

## 🛠 Manual Step-by-Step Mode

If you prefer to run the tools individually, follow these steps.

### 1. Extracting Citations (`agent1_extractor.py`)
Reads base PDFs from `raw/` and uses `pdftotext` to extract a list of citations.
```bash
python agent1_extractor.py
```
**Output:** Generates `extracted_citations.json`.

### 2. Fetching Papers (`agent2_fetcher.py`)
Reads the extracted citations, locates their DOI, and downloads open-access PDF versions.
```bash
python agent2_fetcher.py
```
**Output:** Creates `pulled_pdfs/` directory and `downloaded.json`.

### 3. Ingesting Papers (`agent3_ingestor.py`)
Extracts text and figures from pulled PDFs, uses `gemma4:latest` to describe figures, and inserts everything into ChromaDB and a BM25 index. This step is parallelized for speed.
```bash
# Run with 4 concurrent processes for faster ingestion
python agent3_ingestor.py --workers 4
```
**Output:** Creates/updates `../physics_vectordb` and `../bm25_index.pkl`.

### 4. Interactive Citation Assistant (`agent4_assistant.py`)
Manually query the database for a specific drafted sentence to get a citation suggestion.
```bash
python agent4_assistant.py --text "Recent advances in multilayer graphene nanoribbons have demonstrated tunable impurity scattering properties." --top_k 3
```

### 5. Automated Batch Citer (`agent5_batch_citer.py`)
Process an entire draft file, generating LaTeX citations automatically.
```bash
python agent5_batch_citer.py --file your_draft.txt --out final_draft.txt
```
**Output:** Generates `final_draft.txt` with `\cite{...}` tags, and a `final_draft_citations.json` mapping file.
