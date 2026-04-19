# Citation Agent RAG Pipeline

A four-stage Retrieval-Augmented Generation (RAG) pipeline designed to automate the extraction, fetching, ingestion, and citation-assistance of scientific literature. This system extracts reference strings from a base PDF, automatically downloads open-access versions of the referenced papers, ingests both text and figures into a local Vector Database, and provides context-aware inline citation suggestions for user-written drafts.

## System Architecture

The pipeline consists of four sequential agents:

1. **Agent 1: Extractor (`agent1_extractor.py`)**
   - Parses a base PDF to extract the Reference section.
   - Outputs a list of citation strings.
   
2. **Agent 2: Fetcher (`agent2_fetcher.py`)**
   - Takes extracted citation strings and queries the Crossref API to find the corresponding DOI.
   - Queries the Unpaywall API using the DOI to locate Open Access PDF links.
   - Downloads the PDFs to your local system.
   
3. **Agent 3: Ingestor (`agent3_ingestor.py`)**
   - Reads downloaded PDFs.
   - Uses `layoutparser` (with Detectron2) to separate text blocks from figures/tables.
   - Uses a local Vision-Language Model (`gemma4:latest` via Ollama) to process and caption the extracted figures based on surrounding text context.
   - Embeds text chunks and figure captions using `nomic-embed-text` into a persistent ChromaDB instance.
   - Builds a BM25 index for sparse retrieval.
   
4. **Agent 4: Assistant (`agent4_assistant.py`)**
   - Takes your draft text.
   - Performs a hybrid search (ChromaDB dense vectors + BM25 sparse index) using Reciprocal Rank Fusion (RRF) to find relevant ingested context.
   - Prompts a local LLM (`gemma4:latest`) to suggest how to cite the matched paper accurately and naturally.

## Pre-requisites

### System Requirements
* `pdftotext` (usually available via the `poppler-utils` package on Linux).
* Conda or Miniconda.
* A running instance of `ollama` with the following models pulled:
  * `gemma4:latest`
  * `nomic-embed-text`
* Detectron2 layout model weights downloaded to `../model_final.pth` (relative to the `citation_agent` directory).

### Conda Environment (`rag_prod`)
It is highly recommended to run this pipeline inside a dedicated Conda environment. The code expects to be run in an environment named `rag_prod`. 

See `USAGE.md` for full instructions on setting up the environment, installing dependencies, and running the pipeline through each stage.
