import os
import streamlit as st
import torch
import chromadb
import pickle
import open_clip
import json
import base64
from PIL import Image
from io import BytesIO
import ollama
from sentence_transformers import SentenceTransformer

# --- Constants ---
EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
INDEX_DIR = "index"

# --- Setup Global State ---
@st.cache_resource
def load_models_and_indices():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 2. Setup Persistent ChromaDB Client
    client = chromadb.PersistentClient(path=os.path.join(INDEX_DIR, "chroma_db"))
    collection_visual = client.get_collection(name="multimodal_figures")
    collection_text = client.get_collection(name="semantic_texts")

    # 3. Load Sentence Transformer for Dense Text Search
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 4. Load BM25 Index
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    # 5. Load Source Metadata Map
    metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_v2.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return device, model, tokenizer, collection_visual, collection_text, st_model, bm25, metadata

device, clip_model, clip_tokenizer, chroma_visual, chroma_text, st_model, bm25_index, metadata_list = load_models_and_indices()

# --- Search Implementation ---
def dense_search(query: str, top_k: int = 5):
    text_tokens = clip_tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy().tolist()[0]
        
    results = chroma_visual.query(
        query_embeddings=[text_features],
        n_results=top_k
    )
    
    ranked = []
    if results['ids']:
        ids = results['ids'][0]
        distances = results['distances'][0]
        for doc_id, dist in zip(ids, distances):
            idx = int(doc_id.split('_')[1])
            ranked.append((idx, dist))
    return ranked

def semantic_search(query: str, top_k: int = 5):
    text_embedding = st_model.encode(query).tolist()
    
    results = chroma_text.query(
        query_embeddings=[text_embedding],
        n_results=top_k
    )
    
    ranked = []
    if results['ids']:
        ids = results['ids'][0]
        distances = results['distances'][0]
        for doc_id, dist in zip(ids, distances):
            idx = int(doc_id.split('_')[1])
            ranked.append((idx, dist))
    return ranked

def sparse_search(query: str, top_k: int = 5):
    tokenized_query = query.lower().split()
    doc_scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    
    ranked = []
    for idx in top_indices:
        score = doc_scores[idx]
        if score > 0:
            ranked.append((idx, score))
    return ranked

def hybrid_search(query: str, top_k: int = 5, visual_weight=0.33, semantic_weight=0.33, sparse_weight=0.34):
    k_cand = max(20, top_k * 3)
    dense_results = dense_search(query, top_k=k_cand)
    semantic_results = semantic_search(query, top_k=k_cand)
    sparse_results = sparse_search(query, top_k=k_cand)
    
    rrf_score = {}
    
    for rank, (idx, _) in enumerate(dense_results):
        rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * visual_weight
        
    for rank, (idx, _) in enumerate(semantic_results):
        rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * semantic_weight
        
    for rank, (idx, _) in enumerate(sparse_results):
        rrf_score[idx] = rrf_score.get(idx, 0) + (1.0 / (60 + rank)) * sparse_weight
        
    final_ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for idx, score in final_ranked:
        meta = metadata_list[idx]
        results.append({
            "score": score,
            "metadata": meta
        })
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal RAG Agent", layout="wide", page_icon="🔍")

st.title("🔍 Multimodal RAG Agent")
st.markdown("Search across extracted figures, tables, and plots linearly and semantically.")

# Sidebar Controls
st.sidebar.header("Search Settings")
search_query = st.sidebar.text_input("Enter Search Prompt:", value="misfit function")
top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=20, value=5)

st.sidebar.markdown("### Reciprocal Rank Fusion Weights")
visual_weight = st.sidebar.slider(
    "Visual Match (OpenCLIP)", 
    min_value=0.0, max_value=1.0, value=0.33, step=0.05,
    help="Matches query visually to the cropped plot images."
)
semantic_weight = st.sidebar.slider(
    "Semantic Text Match (MiniLM)", 
    min_value=0.0, max_value=1.0, value=0.33, step=0.05,
    help="Matches query semantically to the generated or extracted captions."
)
sparse_weight = st.sidebar.slider(
    "Lexical Text Match (BM25)", 
    min_value=0.0, max_value=1.0, value=0.34, step=0.05,
    help="Strict keyword matching to the text captions."
)

if st.sidebar.button("Search & Synthesize", type="primary"):
    if search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            results = hybrid_search(search_query, top_k=top_k, 
                                    visual_weight=visual_weight, 
                                    semantic_weight=semantic_weight, 
                                    sparse_weight=sparse_weight)
            
        st.subheader(f"Top {len(results)} Results for '{search_query}'")
            
        if not results:
            st.warning("No results found.")
        else:
            # 1. Show generation
            st.markdown("### ✨ VLM Answer Synthesis")
            
            # Prepare context for VLM
            images_b64 = []
            context_texts = []
            
            for i, res in enumerate(results):
                m = res["metadata"]
                img_path = os.path.join(IMAGES_DIR, m["image_path"])
                
                # Encode Image
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
                        
                # Collect Context
                ctx = m.get("context", "").strip()
                if ctx:
                    context_texts.append(f"[Figure {i+1} from {m.get('doc_name')}]: {ctx}")
                    
            system_prompt = (
                "You are an expert scientific AI assistant. You are provided with a user query, "
                "a set of retrieved relevant scientific figures/plots, and their corresponding textbook/paper captions. "
                "Synthesize a highly accurate, academic answer to the user's query based ONLY on the provided figures and context. "
                "If the answer is not in the figures or context, say so. Do not guess."
            )
            
            user_prompt = f"User Query: {search_query}\n\n"
            if context_texts:
                user_prompt += "Context:\n" + "\n".join(context_texts)
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": images_b64}
            ]
            
            # Stream the answer
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in ollama.chat(model='llama3.2-vision', messages=messages, stream=True):
                        # The chunk dictionary contains a 'message' key with a 'content' field
                        token = chunk['message']['content']
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error communicating with Ollama: {e}")
                    
            st.markdown("---")
            st.markdown("### 📚 Retrieved Sources")
            
            for i, res in enumerate(results):
                m = res["metadata"]
                
                # Setup Display Columns
                col_img, col_txt = st.columns([1, 1.5])
                
                with col_img:
                    img_path = os.path.join(IMAGES_DIR, m["image_path"])
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                        st.image(image, caption=m["image_path"], use_column_width=True)
                    else:
                        st.error(f"Image not found: {m['image_path']}")
                        
                with col_txt:
                    st.markdown(f"### Rank {i+1} (Score: `{res['score']:.4f}`)")
                    st.markdown(f"**Document**: `{m['doc_name']}`")
                    st.markdown(f"**Extracted Entity**: `{m.get('type', 'Figure')}`")
                    st.markdown(f"**Page Number**: `{m.get('page', 'Unknown')}`")
                    
                    with st.expander("Show Document Context / Caption"):
                        context = m.get("context", "None")
                        if not context.strip():
                            st.write("*No associated text context found.*")
                        else:
                            st.text(context)
                
                # Divider between results
                st.markdown("---")
else:
    st.info("Enter a query and click Search on the sidebar.")
