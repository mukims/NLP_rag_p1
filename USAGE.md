# 🛠️ Toolkit Usage Guide

This guide provides instructions for setting up your environment and running both sub-systems in the repository: the **Multimodal Visual RAG** (Streamlit App) and the **Autonomous Citation Pipeline** (Multi-Agent System).

## 0. Prerequisites & Environment Setup

We highly recommend using a dedicated Conda environment named `rag_prod`.

```bash
# 1. Create and activate environment
conda create -n rag_prod python=3.10 -y
conda activate rag_prod

# 2. Install base dependencies
conda install -c conda-forge layoutparser detectron2 sentence-transformers
pip install open_clip_torch chromadb rank_bm25 streamlit ollama pymupdf
pip install opencv-python requests numpy watchdog langchain-ollama langchain-experimental

# 3. System packages (Ubuntu/Debian required for pdftotext)
sudo apt-get update && sudo apt-get install poppler-utils
```

### Models & Weights
1. Ensure your local `ollama` server is running.
2. Pull the required models:
   ```bash
   ollama pull llama3.2-vision
   ollama run gemma4:latest
   ollama pull nomic-embed-text
   ```
3. Place your **Detectron2 (`model_final.pth`)** weights in the root `NLP/` directory.

---

## 🖥️ System 1: Multimodal Visual RAG App (`app.py`)

This system lets you query a corpus of PDFs visually, finding the exact plots or tables relevant to your search.

### Step 1: Extract Figures
Place your target PDFs into `raw/` at the project root, then extract them using Detectron2:
```bash
conda run -n rag_prod python src/extract_detectron2.py
```
*Outputs images to `extracted_data/images/`.*

### Step 2: (Optional) Enrich Captions
Run the Llama 3.2 Vision caption enhancer for better semantic text matching:
```bash
nohup conda run -n rag_prod python src/enrich_metadata.py > enrich.log 2>&1 &
```

### Step 3: Build the Search Index
Embed the figures visually (OpenCLIP), semantically (MiniLM), and lexically (BM25):
```bash
conda run -n rag_prod python src/build_index.py
```

### Step 4: Launch the Streamlit App
```bash
conda run -n rag_prod streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` to start querying your visual database.

---

## 🤖 System 2: Autonomous Citation Agent Pipeline (`citation_agent/`)

This multi-agent system runs entirely inside the `citation_agent/` directory.

### Mode A: Fully Automated (Recommended)
The **Master Orchestrator** monitors directories and runs agents in the background.

```bash
cd citation_agent/
conda activate rag_prod
python master_orchestrator.py
```

**How to use while the orchestrator is running:**
1. **Ingest a paper**: Drop a base PDF into `citation_agent/raw/`. The orchestrator will trigger Agents 1-3 to extract references, fetch open-access PDFs, and ingest them into ChromaDB.
2. **Auto-Cite a draft**: Save your drafted text file (e.g., `my_paper.txt`) into `citation_agent/drafts/`. The orchestrator will instantly trigger Agent 5 and output a fully cited `my_paper_cited.txt` file alongside a JSON mapping of your citations.

### Mode B: Manual Step-by-Step

If you prefer to debug or run specific agents manually, navigate to `citation_agent/` and use:

* **Agent 1 (Extract):** Reads base PDFs from `raw/` and extracts citation strings.
  ```bash
  python agent1_extractor.py
  ```
* **Agent 2 (Fetch):** Downloads open-access PDFs via Crossref/Unpaywall/arXiv.
  ```bash
  python agent2_fetcher.py
  ```
* **Agent 3 (Ingest):** Embeds downloaded papers into the Vector DB (Parallelized).
  ```bash
  python agent3_ingestor.py --workers 4
  ```
* **Agent 4 (Interactive Assistant):** Query for citation suggestions manually.
  ```bash
  python agent4_assistant.py --text "Recent advances show high tunability in nanoribbons." --top_k 3
  ```
* **Agent 5 (Batch Citer):** Process an entire draft file directly.
  ```bash
  python agent5_batch_citer.py --file my_draft.txt --out cited_draft.txt
  ```
