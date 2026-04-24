# 🔬 Multimodal RAG & Autonomous Citation Agent Toolkit

A comprehensive, production-grade toolkit for scientific literature processing. This repository combines **deep-learning layout detection**, **tri-modal hybrid search (Visual + Semantic + Lexical)**, and **autonomous multi-agent workflows** to radically improve how researchers interact with, query, and cite scientific papers.

This project is divided into two primary sub-systems, both powered entirely by **local, privacy-preserving LLMs & VLMs** (via Ollama):

1. **Multimodal Visual RAG System** (Streamlit App)
2. **Autonomous Citation Agent Pipeline** (5-Agent Background Orchestrator)

---

## 🌟 1. Multimodal Visual RAG System (`src/` & `app.py`)

A visual search engine designed to query research papers by their **figures, plots, and tables** — not just text. 

### Key Features
- **Semantic Layout Detection**: Uses Detectron2 (PubLayNet) to semantically extract Figures and Tables with a 20px padding to preserve axis labels.
- **Tri-Modal Hybrid Search (RRF)**: Merges results from OpenCLIP (`ViT-B-32` for visual), MiniLM (`all-MiniLM-L6-v2` for semantic text), and BM25Okapi (lexical matching) using tunable Reciprocal Rank Fusion.
- **VLM Caption Enrichment**: Optionally uses Llama 3.2 Vision to write rich, detailed descriptions for extracted plots, boosting search accuracy.
- **Interactive UI**: A Streamlit application (`app.py`) where you can search, view matched images, and chat with Llama 3.2 Vision about the retrieved charts in real-time.

---

## 🤖 2. Autonomous Citation Agent Pipeline (`citation_agent/`)

A sophisticated, background 5-agent RAG pipeline that automates the tedious parts of writing scientific papers. It extracts reference lists, finds open-access PDFs, ingests their contents into ChromaDB, and automatically adds LaTeX citations to your text drafts.

### The Agents
* **Agent 1 (Extractor)**: Parses base PDFs to extract raw reference strings.
* **Agent 2 (Fetcher)**: Uses Crossref/Unpaywall APIs and arXiv fallbacks to autonomously download open-access PDF references.
* **Agent 3 (Ingestor)**: Parallelized processing to separate text/figures and embed them into ChromaDB using `gemma4:latest` and `nomic-embed-text`.
* **Agent 4 (Assistant)**: Interactive CLI tool to request citation suggestions for a specific draft sentence.
* **Agent 5 (Batch Citer)**: Processes full `.txt` drafts. Uses `gemma4:latest` to identify factual claims and automatically rewrites sentences appending unique LaTeX `\cite{...}` tags, generating a final cited document and bibliography map.

### The Master Orchestrator
The pipeline is fully automated by `master_orchestrator.py`, which uses `watchdog` to silently monitor your filesystem:
- **Drop a PDF** into `citation_agent/raw/` ➡️ Triggers Agents 1-3.
- **Save a text file** in `citation_agent/drafts/` ➡️ Triggers Agent 5 to instantly output a `_cited.txt` version.

---

## 🚀 Getting Started

To install dependencies, download model weights, and run the systems, please see our detailed [USAGE.md](USAGE.md).

---

## 🔑 Key Design Principles

| Design Choice | Rationale |
|--|--|
| **Local Models First** | Both Llama 3.2 Vision and Gemma 4 run locally via Ollama. No API costs, total privacy for unreleased drafts. |
| **Detectron2 > Basic PDF Parsers** | Extracting figures semantically prevents grabbing embedded publisher logos or random vector noise. |
| **Reciprocal Rank Fusion (RRF)** | Safely merges multi-modal scores (CLIP visual + MiniLM semantic + BM25) without scalar conflicts. |
| **Agentic Auto-Citing** | Agent 5 doesn't just match keywords; it explicitly queries an LLM to determine *if* a factual claim warrants a citation first. |

## 📁 Repository Structure

```text
NLP/
├── app.py                      # V1: Streamlit Application (Visual RAG)
├── src/                        # V1: Detectron2 extraction & indexing scripts
├── v1_deployment/              # V1: Standalone deployment bundle
├── model_final.pth             # Required: PubLayNet Detectron2 weights
├── citation_agent/             # V2: The 5-Agent Citation Pipeline
│   ├── master_orchestrator.py  # Automation daemon
│   ├── agent1_extractor.py     
│   ├── agent2_fetcher.py       
│   ├── agent3_ingestor.py      
│   ├── agent4_assistant.py     
│   └── agent5_batch_citer.py   
├── README.md                   # This file
└── USAGE.md                    # Detailed running instructions
```
