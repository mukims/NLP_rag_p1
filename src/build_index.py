import os
import json
import open_clip
import torch
import chromadb
import pickle
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb.config import Settings

EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
METADATA_PATH = os.path.join(EXTRACTED_DIR, "images_metadata_v2.json")

# It might take a bit to extract but we don't want to crash if something failed
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(EXTRACTED_DIR, "images_metadata.json")

INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

def build_index():
    print(f"Loading metadata from {METADATA_PATH}")
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found. Has extraction finished?")
        return
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    
    # We will use open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()

    print(f"Loading SentenceTransformer 'all-MiniLM-L6-v2' on {device}...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print("Initializing ChromaDB...")
    # Chroma DB client
    client = chromadb.PersistentClient(path=os.path.join(INDEX_DIR, "chroma_db"))
    
    # Create or get the multimodal collection (OpenCLIP Visual)
    collection_visual = client.get_or_create_collection(
        name="multimodal_figures",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Create or get the dense text collection (MiniLM Text)
    collection_text = client.get_or_create_collection(
        name="semantic_texts",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Clear existing if any for clean run
    if collection_visual.count() > 0:
        print("Clearing collections and starting over...")
        client.delete_collection("multimodal_figures")
        collection_visual = client.create_collection(
            name="multimodal_figures",
            metadata={"hnsw:space": "cosine"}
        )
    if collection_text.count() > 0:
        client.delete_collection("semantic_texts")
        collection_text = client.create_collection(
            name="semantic_texts",
            metadata={"hnsw:space": "cosine"}
        )

    # BM25 Corpus
    tokenized_corpus = []
    
    print("Embedding data...")
    for idx, item in enumerate(metadata):
        if idx % 50 == 0:
            print(f" Processed {idx}/{len(metadata)}")
            
        img_path = os.path.join(IMAGES_DIR, item["image_path"])
        if not os.path.exists(img_path):
            continue
            
        try:
            # 1. Embed Image
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy().tolist()[0]
                
            # 2. Add Visual Features to ChromaDB
            collection_visual.add(
                ids=[f"fig_{idx}"],
                embeddings=[image_features],
                metadatas=[{
                    "doc_name": item["doc_name"],
                    "image_path": item["image_path"],
                    "context": item.get("context", ""),
                    "page": item.get("page", 0)
                }]
            )
            
            # 3. Dense Text Indexing (MiniLM)
            # Use `rich_context` if VLM enrichment is done, else `context`
            text_context = item.get("rich_context", item.get("context", ""))
            
            if not text_context.strip():
                # If there's literally no text, just fill it with a placeholder so arrays align
                text_context = f"Figure from {item['doc_name']} page {item.get('page', 0)}"
                
            text_embedding = st_model.encode(text_context).tolist()
            
            collection_text.add(
                ids=[f"txt_{idx}"],
                embeddings=[text_embedding],
                metadatas=[{
                    "doc_name": item["doc_name"],
                    "image_path": item["image_path"],
                    "context": text_context,
                    "page": item.get("page", 0)
                }]
            )
            
            # 4. Add to BM25 Corpus
            tokens = text_context.lower().split()
            tokenized_corpus.append(tokens)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Build and save BM25
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
        
    print(f"Indexing complete! Indexed {collection_visual.count()} figures and {collection_text.count()} texts.")

if __name__ == "__main__":
    build_index()
