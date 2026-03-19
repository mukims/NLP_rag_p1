import layoutparser as lp
import cv2
import numpy as np
import fitz
import os

# Monkey patch to ignore the ?dl=1 query in the filename
import detectron2.checkpoint
original_load_file = detectron2.checkpoint.DetectionCheckpointer._load_file
def patched_load_file(self, path):
    if "?dl=1" in path:
        path = path.replace("?dl=1", "")
    return original_load_file(self, path)
detectron2.checkpoint.DetectionCheckpointer._load_file = patched_load_file

def test_detectron():
    print("Loading model...")
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    pdf_path = "raw/p1.pdf"
    print(f"Opening pdf {pdf_path}")
    doc = fitz.open(pdf_path)
    
    for page_idx in range(min(2, len(doc))):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        print(f"Processing page {page_idx}...")
        layout = model.detect(img)
        
        figures = [b for b in layout if b.type in ["Figure", "Table"]]
        print(f"Page {page_idx} found {len(figures)} figures/tables.")
        
if __name__ == "__main__":
    test_detectron()
