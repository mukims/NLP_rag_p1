# Comprehensive Project Report: Multimodal RAG & AI Citation Pipeline

## 🎯 1. Project Objectives

The overarching objective behind this repository is to radically streamline and automate one of the most time-consuming and tedious challenges in scientific research: **finding literature, digesting complex multimodal data (figures, charts, tables), and accurately citing sources while drafting papers.**

### The Problem with Traditional Research Workflows
In modern academia and R&D, researchers spend an exorbitant amount of time managing literature rather than synthesizing new ideas. The typical workflow is highly fragmented and manual:
- **Discovery Bottleneck:** A researcher reads a key paper, spots an interesting claim, and has to manually hunt down the referenced citation across publisher paywalls and preprint servers.
- **The Modality Gap:** Scientific knowledge is predominantly visual. Crucial results are conveyed through plots, micrographs, and data tables. However, traditional text-based search tools completely fail to capture this visual information, leaving a massive gap in how researchers retrieve past findings.
- **Drafting Friction:** Writing a paper involves constant context-switching between drafting text, managing citation keys (e.g., BibTeX), and ensuring that every factual claim is properly backed by the correct source.

### The Problem with Standard RAG Systems
While standard Retrieval-Augmented Generation (RAG) systems have revolutionized text retrieval, they are notoriously poor at handling scientific PDFs. Standard PDF parsers scramble complex multi-column layouts, mangle mathematical equations, and completely ignore images. A standard RAG system cannot answer the question, *"Show me the papers that contain a plot of the graphene bandgap,"* because it only sees text, not layout or visuals.

### The Solution: A Privacy-First, Multimodal Agentic Pipeline
This project was built from the ground up to solve these exact bottlenecks by bridging the gap between computer vision and large language models. The core objectives achieved by this repository are:

1. **Treating Visuals as First-Class Citizens:** By utilizing computer vision layout detection (Detectron2), the system explicitly crops, extracts, and indexes scientific figures and tables. It doesn't just read the paper; it *sees* the paper.
2. **Semantic Visual Understanding:** Leveraging powerful, privacy-preserving local Vision-Language Models (VLMs) like Llama 3.2 Vision to semantically analyze, caption, and describe complex scientific plots so they can be searched alongside standard text.
3. **Automating the Citation Lifecycle:** Building a true multi-agent pipeline that autonomously scrapes reference lists from a seed paper, scours the web (Crossref, Unpaywall, arXiv) to download open-access PDFs, ingests them into a vector database, and then monitors the researcher's local draft.
4. **Agent-Assisted Drafting:** Eliminating the friction of writing by having an AI agent (Gemma 4) read a plain-text draft, identify which factual claims require citations, search the ingested database for the best matching paper, and automatically inject perfect LaTeX `\cite{...}` tags without user intervention.
5. **Ensuring 100% Data Privacy:** By running all inference entirely locally via Ollama, the pipeline guarantees that unreleased, proprietary research drafts and sensitive internal documents are never sent to external corporate APIs.

---

## 🏛️ 2. Architectural Evolution

This repository evolved in two major phases, transforming from a search engine into an active, multi-agent assistant.

### Phase 1: Multimodal Visual RAG (V1)
The initial goal was to build a visual search engine for scientific figures. 
- **Core Mechanism**: Uses `layoutparser` and Detectron2 (`PubLayNet` weights) to semantically crop figures from PDFs, adding strategic padding to preserve axis labels.
- **Tri-Modal Indexing**: Embeds data across three dimensions: visually (via OpenCLIP), semantically (MiniLM text embeddings of VLM-enriched captions), and lexically (BM25).
- **Interface**: A Streamlit application (`app.py`) allowing users to query topics and instantly see matching charts alongside Llama 3.2 Vision's analysis.

### Phase 2: Autonomous Multi-Agent Citation Pipeline (V2)
The project then expanded into a fully autonomous workflow to directly assist paper writing. The `citation_agent/` directory houses **5 specialized AI agents** orchestrated by a background daemon.
- **Agent 1 (Extractor)**: Reads base PDFs and extracts their raw reference lists.
- **Agent 2 (Fetcher)**: Cross-references those citations with Crossref/Unpaywall/arXiv APIs to download open-access PDFs locally.
- **Agent 3 (Ingestor)**: A high-performance, parallelized ingestor that chunks the PDFs, processes figures with `gemma4:latest`, and stores them in ChromaDB.
- **Agent 4 & 5 (Assistant & Batch Citer)**: AI agents that read the user's plain-text draft, identify factual claims that require backing, perform hybrid database searches, and output the draft with accurate `\cite{...}` LaTeX tags without any human intervention.

---

## 🚀 3. How to Use This Repository

### System Setup
The pipeline relies heavily on the `rag_prod` Conda environment to manage complex dependencies like Detectron2 and ChromaDB.
1. Create the environment: `conda create -n rag_prod python=3.10`
2. Install dependencies (see `USAGE.md` for exact pip/conda commands).
3. Ensure the local Ollama server is running with `llama3.2-vision`, `gemma4:latest`, and `nomic-embed-text` pulled.
4. Download the required Detectron2 weights (`model_final.pth`).

### Workflow A: Visual Search (Streamlit UI)
If you want to visually search across ingested PDFs:
1. Place raw PDFs in `/raw/`.
2. Run `src/extract_detectron2.py` to extract images.
3. Run `src/build_index.py` to embed the data into the local vector databases.
4. Launch the UI: `streamlit run app.py`

### Workflow B: Autonomous Drafting (Agent Pipeline)
If you are drafting a paper and want the AI to handle your bibliography and citations:
> [!TIP]
> This is the recommended, fully hands-off mode.

1. **Start the Orchestrator**: Run `python master_orchestrator.py` inside the `citation_agent/` directory. This daemon will monitor your folders indefinitely.
2. **Feed the Pipeline**: Drop any PDF you find interesting into `citation_agent/raw/`. The orchestrator will automatically extract its references, download the corresponding papers, and index them.
3. **Draft and Auto-Cite**: Write your paper in plain text and save it to `citation_agent/drafts/my_draft.txt`. The orchestrator will immediately detect the save, trigger Agent 5, and output `my_draft_cited.txt` containing perfect LaTeX citations.

---

## 💡 4. Key Takeaways
This repository represents a fundamental shift from **passive retrieval** (standard RAG) to **active agency**. By combining deterministic tools (APIs, layout parsers, file watchers) with generative reasoning (`gemma4` via Ollama), it successfully shifts the burden of literature management and citation formatting entirely onto the machine, while maintaining **100% data privacy** via local model inference.
