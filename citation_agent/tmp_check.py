import os, json, re

images_dir = '/home/shardul/NLP/extracted_data/images'
images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
if not images:
    print('No images found.')
    exit()

latest_img = max(images, key=lambda x: os.path.getmtime(os.path.join(images_dir, x)))
match = re.search(r'(.*)_p\d+_f\d+\.png$', latest_img)

if match:
    pdf_name = match.group(1)
    try:
        with open('downloaded.json') as f:
            paths = list(json.load(f).keys())
        
        for i, p in enumerate(paths):
            if os.path.basename(p).strip().replace(' ', '_').lower() == pdf_name:
                print(f'Latest image generated for: {pdf_name}')
                print(f'Currently around file {i+1} of {len(paths)}.')
                print(f'Files left to process: {len(paths) - i - 1}')
                break
    except Exception as e:
        print(f"Error reading JSON: {e}")
