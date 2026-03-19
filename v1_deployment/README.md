# Scientific Multimodal VLM RAG (Version 1.0)

> **Domain Application:** This system is specifically designed to parse and retrieve complex scientific literature—such as inverse problems in quantum physics, calculating conductance spectra in disordered two-terminal devices (like Graphene Nanoribbons), and analyzing structural and compositional information from structural simulations (Tight-Binding/DFT).

This directory contains the finalized, completely standalone Version 1.0 implementation of the multimodal retrieval system. It has been stripped of experimental files and features heavily commented code for human readability.

## Flow of Operations

1. **Extraction** (`extract_images.py`):
   - Reads PDFs from `raw/`.
   - Uses Detectron2 (Deep Learning Object Detection) to slice out high-quality figures/tables.
   - Padds bounding boxes to preserve axis labels.
   - Saves cropped figures to `extracted_data/images`.
   - Uses PyMuPDF to extract text from the page and algorithmically matches the caption to the figure using spatial proximity.
   - Saves all mapping data to `extracted_data/images_metadata_d2.json`.

2. **Indexing** (`build_index.py`):
   - Iterates through the extracted images and text.
   - Computes dense mathematical vision embeddings for every image using `OpenCLIP` (`ViT-B-32`).
   - Stores these visual features persistently in **ChromaDB** (`index/chroma_db`).
   - Tokenizes all extracted text/captions and builds a **BM25** lexical index (`index/bm25_index.pkl`).

3. **Application & Retrieval Interface** (`retriever_app.py`):
   - A `Streamlit` web interface that the user interacts with.
   - **Hybrid Retrieval**: It takes user text queries and queries *both* ChromaDB (visually) and BM25 (textually). It merges their results using Reciprocal Rank Fusion to yield the best matches.
   - **VLM Generation**: It takes the top matched figures (base64 encoded), their textual captions, and the user's explicit query, and streams them into **Llama 3.2 Vision** via Ollama to generate a synthesized conversational answer.

## Usage

If you already have the data indexed:
```bash
streamlit run app.py
```

If you want to run the pipeline from scratch:
```bash
python extract_images.py
python build_index.py
streamlit run app.py
```
