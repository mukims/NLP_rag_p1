import pymupdf4llm
import os
import json

def test_extract():
    pdf_path = "raw/p1.pdf"
    if not os.path.exists(pdf_path):
        pdfs = [f for f in os.listdir("raw") if f.endswith(".pdf")]
        if not pdfs:
            print("No pdf found")
            return
        pdf_path = os.path.join("raw", pdfs[0])
        
    print(f"Extracting from {pdf_path}")
    os.makedirs("test_out", exist_ok=True)
    
    # Extract markdown and images
    md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path="test_out")
    
    with open("test_out/output.md", "w") as f:
        f.write(md_text)
        
    print("Done")

if __name__ == "__main__":
    test_extract()
