"""
Multimodal RAG - Step 1: Image Extraction

This script is responsible for finding and cropping scientific figures and tables 
from raw PDF files. It uses a Deep Learning object detection model (Detectron2) 
to locate the figures, and PyMuPDF to extract the nearby text to use as "captions".

Dependencies:
    - layoutparser (with detectron2 backend)
    - PyMuPDF (fitz)
    - OpenCV (cv2)
"""

import os
import json
import cv2
import fitz  # PyMuPDF
import numpy as np
import layoutparser as lp

# --- Paths Configuration ---
# All PDFs should be placed in this folder before running
RAW_DIR = "raw"

# Output directories for cropped figures and the metadata JSON mapping
EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def patch_detectron_loader():
    """
    Monkey patch to fix a known bug in `fvcore` (a detectron2 dependency) 
    where it crashes if the model weights file has a URL query string (like ?dl=1).
    """
    import detectron2.checkpoint
    original_load_file = detectron2.checkpoint.DetectionCheckpointer._load_file
    
    def patched_load_file(self, path):
        if "?dl=1" in path:
            path = path.replace("?dl=1", "")
        return original_load_file(self, path)
        
    detectron2.checkpoint.DetectionCheckpointer._load_file = patched_load_file


def find_caption_for_box(fig_box_img, text_blocks_img, box_type):
    """
    This function tries to find the text that belongs to a specific figure/table.
    It works by calculating the physical distance between the figure's bounding box 
    and all the text blocks on the page.
    
    Args:
        fig_box_img: Tuples of (x1, y1, x2, y2) coords for the figure box
        text_blocks_img: List of dictionaries containing text blocks and their coords
        box_type: 'Figure' or 'Table'
        
    Returns:
        The best matching text string (the caption), or an empty string if nothing matches.
    """
    x1, y1, x2, y2 = fig_box_img
    
    # Filter out empty or very short blocks of text
    valid_blocks = [b for b in text_blocks_img if len(b['text'].strip()) > 5]
    
    best_block = ""
    min_dist = float('inf')
    
    for b in valid_blocks:
        tx0, ty0, tx1, ty1 = b['bbox']
        text = b['text']
        
        # Calculate vertical distances
        dist_below = ty0 - y2  # Distance from bottom of figure to top of text
        dist_above = y1 - ty1  # Distance from top of figure to bottom of text
        
        # Calculate horizontal overlap to avoid grabbing text from an adjacent column
        horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
        if horiz_overlap < 10 and (x2 - x1 > 100):
            continue  # Likely in a totally different column
            
        # Figures usually have captions *below* them
        if box_type == "Figure" and dist_below > -20 and dist_below < 400:
            score = dist_below
            if text.strip().lower().startswith("fig"):
                # Massive bonus if the text block actually starts with "Fig"
                score -= 200
            
            if score < min_dist:
                min_dist = score
                best_block = text
                
        # Tables usually have titles *above* them
        elif box_type == "Table" and dist_above > -20 and dist_above < 400:
            score = dist_above
            if text.strip().lower().startswith("table"):
                score -= 200
                
            if score < min_dist:
                min_dist = score
                best_block = text
                
    return best_block


def main():
    patch_detectron_loader()
    
    print("Loading Detectron2 Layout Model...")
    # This points to our pre-downloaded Detectron2 LayoutParser weights trained on PubLayNet
    local_weights = os.path.abspath("model_final.pth")
    if not os.path.exists(local_weights):
        raise FileNotFoundError(f"Missing {local_weights}. Please download the LayoutParser weights.")
        
    # Initialize the deep learning layout model
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        model_path=local_weights,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], # 50% confidence threshold
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    all_metadata = []
    
    # Loop through every single PDF in our raw files directory
    for filename in os.listdir(RAW_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(RAW_DIR, filename)
        doc_name = os.path.splitext(filename)[0]
        print(f"Processing {filename}...")
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # We want high quality images for indexing so the AI can read them (200 DPI)
            dpi = 200
            zoom = dpi / 72.0  # Calculate zoom factor from the default 72 DPI
            
            # Iterate through all pages of this PDF
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                
                # 1. Convert the PDF page into an image array for Detectron
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                # Convert colors properly based on if there is an alpha channel
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                # 2. Ask Detectron2 to locate all elements on the page image
                layout = model.detect(img)
                
                # 3. Extract all text blocks on the page using PyMuPDF (much faster than OCR)
                raw_blocks = page.get_text("blocks")
                text_blocks_img = []
                for b in raw_blocks:
                    # b[6] == 0 means this block is text (as opposed to an image block)
                    if b[6] == 0: 
                        # Scale the PyMuPDF coords to match our 200 DPI image coords
                        tx0, ty0, tx1, ty1 = [c * zoom for c in b[:4]]
                        text = b[4].replace('\n', ' ')
                        text_blocks_img.append({
                            'bbox': (tx0, ty0, tx1, ty1),
                            'text': text
                        })
                
                # 4. Filter only for Figures and Tables
                figures = [b for b in layout if b.type in ["Figure", "Table"]]
                
                # Process each figure/table found on this page
                for i, fig in enumerate(figures):
                    # We pad the bounding box by 20 pixels.
                    # This prevents tightly cropped figures from cutting off the X/Y axis labels!
                    pad = 20
                    x1, y1, x2, y2 = fig.coordinates
                    x1 = max(0, int(x1 - pad))
                    y1 = max(0, int(y1 - pad))
                    x2 = min(img.shape[1], int(x2 + pad))
                    y2 = min(img.shape[0], int(y2 + pad))
                    
                    # Crop the image array
                    cropped_img = img[y1:y2, x1:x2]
                    
                    # Save the cropped figure to disk
                    out_name = f"{doc_name}_p{page_idx}_f{i}.png"
                    out_path = os.path.join(IMAGES_DIR, out_name)
                    cv2.imwrite(out_path, cropped_img)
                    
                    # 5. Attempt to find the text caption for this specific figure
                    context = find_caption_for_box((x1, y1, x2, y2), text_blocks_img, fig.type)
                    
                    # Save all properties so we can index them later
                    all_metadata.append({
                        "doc_name": doc_name,
                        "image_path": out_name,
                        "context": context,
                        "page": page_idx,
                        "type": fig.type
                    })
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Finally, save the comprehensive mapping of every image to its context
    metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=4)
        
    print(f"\nExtraction complete! Over {len(all_metadata)} figures saved to {IMAGES_DIR}.")
    print(f"Metadata map saved to {metadata_path}")


if __name__ == "__main__":
    main()
