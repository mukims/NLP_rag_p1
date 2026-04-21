# Citation Agent RAG Pipeline

A comprehensive, fully automated Retrieval-Augmented Generation (RAG) pipeline designed to automate the extraction, fetching, ingestion, and citation-assistance of scientific literature using local models. 

This system extracts reference strings from a base PDF, automatically downloads open-access versions of the referenced papers, ingests both text and figures into a local Vector Database, and automatically injects proper LaTeX citations into your text drafts without duplication.

## System Architecture

The pipeline consists of five sequential agents, managed by an intelligent background orchestrator.

### The Master Orchestrator (`master_orchestrator.py`)
- Continuously monitors the `raw/` directory for new PDFs and the `drafts/` directory for text drafts.
- Automatically handles batching and cooldowns to prevent GPU memory crashes.
- Drives the entire pipeline (Agents 1-3, and Agent 5) completely hands-off.

### Data Ingestion (Agents 1-3)
1. **Agent 1: Extractor (`agent1_extractor.py`)**
   - Parses a base PDF to extract the Reference section.
   - Outputs a list of citation strings.
   
2. **Agent 2: Fetcher (`agent2_fetcher.py`)**
   - Takes extracted citation strings and queries the Crossref API to find the corresponding DOI.
   - Queries the Unpaywall API using the DOI to locate Open Access PDF links.
   - Falls back to arXiv search if no open access link is found.
   - Downloads the PDFs to your local system.
   
3. **Agent 3: Ingestor (`agent3_ingestor.py`)**
   - Parallelized multimodal processing of downloaded PDFs.
   - Uses `layoutparser` (with Detectron2) to separate text blocks from figures/tables.
   - Uses a local Vision-Language Model (`gemma4:latest` via Ollama) to process and caption the extracted figures based on surrounding text context.
   - Embeds text chunks and figure captions using `nomic-embed-text` into a persistent ChromaDB instance.
   - Builds a BM25 index for sparse retrieval.

### Inference & Writing (Agents 4-5)
4. **Agent 4: Assistant (`agent4_assistant.py`)**
   - Interactive CLI assistant. Takes a short snippet of your draft text.
   - Performs a hybrid search (ChromaDB dense vectors + BM25 sparse index) using Reciprocal Rank Fusion (RRF).
   - Prompts `gemma4:latest` to suggest how to cite the matched paper accurately and naturally in your snippet.

5. **Agent 5: Batch Citer (`agent5_batch_citer.py`)**
   - Automated draft processor. Takes a full `.txt` draft document.
   - Semantically processes sentences, querying `gemma4` to decide if factual claims require citations.
   - Retrieves contexts and automatically rewrites sentences appending unique LaTeX `\cite{cite_key}` tags.
   - Generates a fully cited document and a JSON citation bibliography map, ensuring zero duplicate cite keys.

## Pre-requisites

### System Requirements
* `pdftotext` (usually available via the `poppler-utils` package on Linux).
* Conda or Miniconda.
* A running instance of `ollama` with the following models pulled:
  * `gemma4:latest`
  * `nomic-embed-text`
* Detectron2 layout model weights downloaded to `../model_final.pth` (relative to the `citation_agent` directory).
* `watchdog` Python library for the orchestrator.

### Conda Environment (`rag_prod`)
It is highly recommended to run this pipeline inside a dedicated Conda environment. The code expects to be run in an environment named `rag_prod`. 

See `USAGE.md` for full instructions on setting up the environment, configuring the orchestrator, and running the pipeline through each stage.
