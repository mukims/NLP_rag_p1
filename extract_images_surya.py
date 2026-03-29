"""
Multimodal RAG - Step 1: Image Extraction (Surya Version)

Extracts figures and tables from PDF research articles using Surya layout detection.
Works natively on Mac M2 via Metal (MPS) — no detectron2 needed.

Dependencies:
    pip install surya-ocr pymupdf pillow tqdm
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from surya.layout import LayoutPredictor, FoundationPredictor

# --- Paths ---
RAW_DIR = "raw"
EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Labels Surya uses for figures/tables
FIGURE_LABELS = {"Picture", "Figure", "Image"}
TABLE_LABELS  = {"Table"}
TARGET_LABELS = FIGURE_LABELS | TABLE_LABELS


def find_caption(page, bbox, box_type, zoom):
    """
    Find the nearest caption text for a detected figure or table.
    - Figures: caption is usually *below* the figure, starting with 'Fig'
    - Tables:  caption is usually *above* the table, starting with 'Table'
    """
    x1, y1, x2, y2 = bbox
    raw_blocks = page.get_text("blocks")
    best, min_dist = "", float("inf")

    for b in raw_blocks:
        if b[6] != 0:  # skip non-text blocks
            continue
        tx0, ty0, tx1, ty1 = [c * zoom for c in b[:4]]
        text = b[4].replace("\n", " ").strip()

        if len(text) < 5:
            continue

        # Ignore text in a completely different column
        horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
        if horiz_overlap < 10 and (x2 - x1) > 100:
            continue

        if box_type == "Figure":
            dist = ty0 - y2  # distance below the figure
            if -20 < dist < 400:
                score = dist - (200 if text.lower().startswith("fig") else 0)
                if score < min_dist:
                    min_dist, best = score, text

        elif box_type == "Table":
            dist = y1 - ty1  # distance above the table
            if -20 < dist < 400:
                score = dist - (200 if text.lower().startswith("table") else 0)
                if score < min_dist:
                    min_dist, best = score, text

    return best


def main():
    pdfs = list(Path(RAW_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in '{RAW_DIR}/'")
        return

    print(f"Found {len(pdfs)} PDF(s). Loading Surya layout model...")
    foundation = FoundationPredictor()       # loads base model (uses MPS on M2)
    predictor = LayoutPredictor(foundation)  # build layout predictor on top
    print("Model loaded.\n")

    all_metadata = []
    DPI = 150
    ZOOM = DPI / 72.0
    PAD = 20  # pixel padding around each crop

    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        doc_name = pdf_path.stem
        print(f"\n→ {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"  Failed to open: {e}")
            continue

        for page_idx in tqdm(range(len(doc)), desc=f"  Pages", leave=False):
            page = doc[page_idx]

            # Convert PDF page → PIL Image for Surya
            pix = page.get_pixmap(dpi=DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Run layout detection
            try:
                predictions = predictor([img])
            except Exception as e:
                print(f"  Skipping page {page_idx}: {e}")
                continue

            bboxes = predictions[0].bboxes

            figure_count = 0
            for box in bboxes:
                if box.label not in TARGET_LABELS:
                    continue

                box_type = "Table" if box.label in TABLE_LABELS else "Figure"

                # Apply padding, clamp to image bounds
                x1, y1, x2, y2 = [int(c) for c in box.bbox]
                x1 = max(0, x1 - PAD)
                y1 = max(0, y1 - PAD)
                x2 = min(img.width,  x2 + PAD)
                y2 = min(img.height, y2 + PAD)

                # Skip tiny detections (noise)
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                # Crop and save
                cropped = img.crop((x1, y1, x2, y2))
                out_name = f"{doc_name}_p{page_idx}_f{figure_count}.png"
                out_path = os.path.join(IMAGES_DIR, out_name)
                cropped.save(out_path)

                # Find caption from surrounding text
                caption = find_caption(page, (x1, y1, x2, y2), box_type, ZOOM)

                all_metadata.append({
                    "doc_name": doc_name,
                    "image_path": out_name,
                    "context": caption,
                    "page": page_idx,
                    "type": box_type
                })

                figure_count += 1

            if figure_count:
                print(f"  Page {page_idx}: {figure_count} figure(s) found")

        doc.close()

    # Save metadata
    metadata_path = os.path.join(EXTRACTED_DIR, "images_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"\n✅ Done! {len(all_metadata)} figures/tables extracted.")
    print(f"   Images → {IMAGES_DIR}/")
    print(f"   Metadata → {metadata_path}")


if __name__ == "__main__":
    main()
