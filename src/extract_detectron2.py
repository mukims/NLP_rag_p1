import os
import json
import cv2
import fitz
import numpy as np
import layoutparser as lp

RAW_DIR = "raw"
EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Monkey patch fvcore to ignore ?dl=1
import detectron2.checkpoint
original_load_file = detectron2.checkpoint.DetectionCheckpointer._load_file
def patched_load_file(self, path):
    if "?dl=1" in path:
        path = path.replace("?dl=1", "")
    return original_load_file(self, path)
detectron2.checkpoint.DetectionCheckpointer._load_file = patched_load_file

def find_caption_for_box(fig_box_img, text_blocks_img, box_type):
    x1, y1, x2, y2 = fig_box_img
    valid_blocks = [b for b in text_blocks_img if len(b['text'].strip()) > 5]
    
    best_block = ""
    min_dist = float('inf')
    
    for b in valid_blocks:
        tx0, ty0, tx1, ty1 = b['bbox']
        text = b['text']
        
        dist_below = ty0 - y2
        dist_above = y1 - ty1
        
        # Find horizontal overlap to ensure it's not a caption from another column
        horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
        if horiz_overlap < 10 and (x2 - x1 > 100):
            # Probably in a different column
            continue
            
        if box_type == "Figure" and dist_below > -20 and dist_below < 400:
            score = dist_below
            if text.strip().lower().startswith("fig"):
                score -= 200
            
            if score < min_dist:
                min_dist = score
                best_block = text
                
        elif box_type == "Table" and dist_above > -20 and dist_above < 400:
            score = dist_above
            if text.strip().lower().startswith("table"):
                score -= 200
                
            if score < min_dist:
                min_dist = score
                best_block = text
    
    if not best_block:
        # Fallback
        min_dist = float('inf')
        for b in valid_blocks:
            tx0, ty0, tx1, ty1 = b['bbox']
            text = b['text']
            # Only consider blocks with horizontal overlap
            horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
            if horiz_overlap < 10 and (x2 - x1 > 100):
                continue
                
            dist_below = max(0, ty0 - y2)
            dist_above = max(0, y1 - ty1)
            dist = min(dist_below, dist_above)
            
            if dist < min_dist:
                min_dist = dist
                best_block = text
                
    return best_block

def main():
    print("Loading Detectron2 Layout Model...")
    # Make sure to point to our local downloaded weights
    local_weights = os.path.abspath("model_final.pth")
    if not os.path.exists(local_weights):
        raise FileNotFoundError(f"Missing {local_weights}")
        
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        model_path=local_weights,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    all_metadata = []
    
    for filename in os.listdir(RAW_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(RAW_DIR, filename)
        doc_name = os.path.splitext(filename)[0]
        print(f"Processing {filename}...")
        
        try:
            doc = fitz.open(pdf_path)
            # Default PyMuPDF dpi is 72. 
            dpi = 200
            zoom = dpi / 72.0
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                layout = model.detect(img)
                
                # Get PyMuPDF text blocks and scale their coordinates to match the image
                raw_blocks = page.get_text("blocks")
                text_blocks_img = []
                for b in raw_blocks:
                    if b[6] == 0: # 0 means text block
                        # Scale coords
                        tx0, ty0, tx1, ty1 = [c * zoom for c in b[:4]]
                        text = b[4].replace('\n', ' ')
                        text_blocks_img.append({
                            'bbox': (tx0, ty0, tx1, ty1),
                            'text': text
                        })
                
                figures = [b for b in layout if b.type in ["Figure", "Table"]]
                
                for i, fig in enumerate(figures):
                    # Pad box by 20 pixels to ensure axis labels aren't cut off
                    pad = 20
                    x1, y1, x2, y2 = fig.coordinates
                    x1 = max(0, int(x1 - pad))
                    y1 = max(0, int(y1 - pad))
                    x2 = min(img.shape[1], int(x2 + pad))
                    y2 = min(img.shape[0], int(y2 + pad))
                    
                    cropped_img = img[y1:y2, x1:x2]
                    
                    out_name = f"{doc_name}_p{page_idx}_f{i}.png"
                    out_path = os.path.join(IMAGES_DIR, out_name)
                    cv2.imwrite(out_path, cropped_img)
                    
                    # Associate context
                    context = find_caption_for_box((x1, y1, x2, y2), text_blocks_img, fig.type)
                    
                    all_metadata.append({
                        "doc_name": doc_name,
                        "image_path": out_name,
                        "context": context,
                        "page": page_idx,
                        "type": fig.type
                    })
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=4)
        
    print(f"Extraction complete! Saved {len(all_metadata)} items using Detectron2.")

if __name__ == "__main__":
    main()
