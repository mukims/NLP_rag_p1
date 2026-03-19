import os
import torch
import chromadb
import pickle
import open_clip
import json
from sentence_transformers import SentenceTransformer

EXTRACTED_DIR = "extracted_data"
INDEX_DIR = "index"

class HybridRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ViT-B-32 model on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=os.path.join(INDEX_DIR, "chroma_db"))
        self.collection_visual = self.client.get_collection(name="multimodal_figures")
        self.collection_text = self.client.get_collection(name="semantic_texts")
        
        print(f"Loading SentenceTransformer 'all-MiniLM-L6-v2' on {self.device}...")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        print("Loading BM25 Index...")
        bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_v2.json")
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata.json")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def dense_search(self, query: str, top_k: int = 5):
        # Embed text query
        text_tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().tolist()[0]
            
        # Query ChromaDB (returns nearest images)
        results = self.collection_visual.query(
            query_embeddings=[text_features],
            n_results=top_k
        )
        
        # Format results: list of (doc_id, score/distance)
        ranked = []
        if results['ids']:
            ids = results['ids'][0]
            distances = results['distances'][0]
            for doc_id, dist in zip(ids, distances):
                # Normalize doc_id from fig_X to X
                idx = int(doc_id.split('_')[1])
                ranked.append((idx, dist))
        return ranked

    def semantic_search(self, query: str, top_k: int = 5):
        # Embed text query
        text_embedding = self.st_model.encode(query).tolist()
            
        # Query ChromaDB (returns nearest text captions)
        results = self.collection_text.query(
            query_embeddings=[text_embedding],
            n_results=top_k
        )
        
        # Format results: list of (doc_id, score/distance)
        ranked = []
        if results['ids']:
            ids = results['ids'][0]
            distances = results['distances'][0]
            for doc_id, dist in zip(ids, distances):
                # Normalize doc_id from txt_X to X
                idx = int(doc_id.split('_')[1])
                ranked.append((idx, dist))
        return ranked
        
    def sparse_search(self, query: str, top_k: int = 5):
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and get top K
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        ranked = []
        for idx in top_indices:
            score = doc_scores[idx]
            if score > 0:
                ranked.append((idx, score))
                
        return ranked

    def hybrid_search(self, query: str, top_k: int = 3, visual_weight=0.33, semantic_weight=0.33, sparse_weight=0.34):
        """Combines BM25, CLIP, and MiniLM using Tri-Modal Reciprocal Rank Fusion (RRF)"""
        k_cand = max(20, top_k * 3)
        dense_results = self.dense_search(query, top_k=k_cand)
        semantic_results = self.semantic_search(query, top_k=k_cand)
        sparse_results = self.sparse_search(query, top_k=k_cand)
        
        # RRF
        rrf_score = {}
        for rank, (idx, _) in enumerate(dense_results):
            # Chroma returns distance (lower is better), so rank 0 is best
            rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * visual_weight
            
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * semantic_weight
            
        for rank, (idx, _) in enumerate(sparse_results):
            # BM25 returns score (higher is better), already sorted desc
            rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * sparse_weight
            
        # Sort by RRF score descending
        final_ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Map back to full metadata
        final_results = []
        for idx, score in final_ranked:
            meta = self.metadata[idx]
            final_results.append({
                "score": score,
                "metadata": meta
            })
            
        return final_results

def test_retriever():
    print("Initializing Retriever...")
    retriever = HybridRetriever()
    
    queries = [
        "misfit function",
        "diagram of a generic framework connecting multiple physical systems",
        "ratio of electrical to thermal conductivity"
    ]
    
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        results = retriever.hybrid_search(q, top_k=2)
        for i, res in enumerate(results):
            m = res['metadata']
            print(f"Rank {i+1} (Score: {res['score']:.4f})")
            print(f"  File: {m['image_path']}")
            print(f"  Doc: {m['doc_name']} (Page {m.get('page', '?')})")
            context = m.get('context', '')
            if len(context) > 100:
                context = context[:100] + "..."
            print(f"  Context preview: {context}")

if __name__ == "__main__":
    test_retriever()
