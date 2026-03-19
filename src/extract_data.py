import os
import json
import pymupdf4llm
import re

RAW_DIR = "raw"
EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
DOCS_DIR = os.path.join(EXTRACTED_DIR, "docs")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def parse_markdown_for_images(md_text, doc_name):
    # Split the markdown into chunks based on double newlines
    chunks = re.split(r'\n\s*\n', md_text)
    
    image_metadata = []
    
    for i, chunk in enumerate(chunks):
        # Find all markdown images in this chunk
        images = re.findall(r'!\[.*?\]\((.*?\.png)\)', chunk)
        if not images:
            images = re.findall(r'!\[.*?\]\((.*?\.jpe?g)\)', chunk)
            
        for img_path in images:
            # We want to get context for this image.
            # Usually the caption is immediately after or before the image chunk.
            
            # Combine 1 chunk before, current, and 2 chunks after as the "caption context"
            start_idx = max(0, i - 1)
            end_idx = min(len(chunks), i + 3) # Up to 2 chunks after
            
            context = "\n\n".join(chunks[start_idx:end_idx])
            
            # Additional heuristic: try to isolate the exact caption "FIG. X" or "Figure X"
            # It's better to just give the hybrid retriever the full surrounding context
            caption_clean = re.sub(r'!\[.*?\]\((.*?)\)', '', context).strip()
            
            image_metadata.append({
                "doc_name": doc_name,
                "image_path": os.path.basename(img_path),
                "context": caption_clean, # Used for BM25 and Text Embedding
                "chunk_index": i
            })
            
    return image_metadata


def main():
    if not os.path.exists(RAW_DIR):
        print(f"Directory '{RAW_DIR}' not found.")
        return

    all_image_metadata = []
    
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            doc_name = os.path.splitext(filename)[0]
            print(f"Extracting {filename}...")
            
            try:
                # Extract markdown with embedded images
                md_text = pymupdf4llm.to_markdown(
                    pdf_path, 
                    write_images=True, 
                    image_path=IMAGES_DIR,
                    dpi=150 # Decent resolution for OpenCLIP later
                )
                
                # Save the raw markdown doc
                md_path = os.path.join(DOCS_DIR, f"{doc_name}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_text)
                    
                # Extract image metadata and context
                img_meta = parse_markdown_for_images(md_text, doc_name)
                all_image_metadata.extend(img_meta)
                
                print(f"  Found {len(img_meta)} images/figures.")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    # Save the consolidated metadata
    metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_image_metadata, f, indent=4)
        
    print(f"Extraction complete. Total images: {len(all_image_metadata)}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
