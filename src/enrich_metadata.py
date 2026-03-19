import json
import os
import base64
import ollama
from tqdm import tqdm

EXTRACTED_DIR = "extracted_data"
IMAGES_DIR = os.path.join(EXTRACTED_DIR, "images")
INPUT_METADATA = os.path.join(EXTRACTED_DIR, "images_metadata_d2.json")
OUTPUT_METADATA = os.path.join(EXTRACTED_DIR, "images_metadata_v2.json")

def generate_rich_caption(image_path, base_context):
    full_path = os.path.join(IMAGES_DIR, image_path)
    if not os.path.exists(full_path):
        return base_context
        
    with open(full_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = (
        "You are an expert scientific researcher. Analyze this figure/table from a scientific paper. "
        "Describe exactly what this figure shows in deep technical and semantic detail. "
        "Mention the axes, variables, trends, and the core scientific concept being illustrated. "
        "If there is a provided original caption, incorporate its meaning."
    )
    if base_context:
        prompt += f"\n\nOriginal Paper Caption: {base_context}"
        
    messages = [
        {"role": "user", "content": prompt, "images": [img_b64]}
    ]
    
    try:
        response = ollama.chat(model='llama3.2-vision', messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama for {image_path}: {e}")
        return base_context

def main():
    if not os.path.exists(INPUT_METADATA):
        print(f"Input metadata not found: {INPUT_METADATA}")
        return
        
    with open(INPUT_METADATA, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print(f"Enriching {len(metadata)} items with Llama 3.2 Vision...")
    
    # We'll save incrementally to avoid losing progress
    enriched_metadata = []
    
    # To save time in testing, let's just do it for all if we can, but it might take a while.
    # We will process all of them.
    for i, item in enumerate(tqdm(metadata)):
        try:
            rich_context = generate_rich_caption(item['image_path'], item.get('context', ''))
            
            enrich_item = item.copy()
            enrich_item['rich_context'] = rich_context
            enriched_metadata.append(enrich_item)
            
            # Save every 10 items
            if (i + 1) % 10 == 0:
                with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
                    json.dump(enriched_metadata, f, indent=4)
                    
        except Exception as e:
            print(f"Failed to enrich {item['image_path']}: {e}")
            enriched_metadata.append(item)
            
    # Final save
    with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
        json.dump(enriched_metadata, f, indent=4)
        
    print("VLM Enrichment Complete! Saved to", OUTPUT_METADATA)

if __name__ == "__main__":
    main()
