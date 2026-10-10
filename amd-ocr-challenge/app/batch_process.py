import os
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from app.config import SERVER_URL

def process_single_image(image_path: Path, output_dir: Path, root_input_dir: Path):
    """
    Sends a single image to the OCR server and saves the result.
    """
    try:
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            response = requests.post(SERVER_URL, files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()

            # Create a unique filename based on relative path to avoid collisions
            # e.g., train/image1.jpg -> train_image1_output.json
            relative_path = image_path.relative_to(root_input_dir)
            unique_name = str(relative_path).replace(os.sep, '_').replace('.', '_')
            output_filename = f"{unique_name}_output.json"
            output_path = output_dir / output_filename

            with open(output_path, 'w') as out_f:
                json.dump(result, out_f, indent=2)
            return True, image_path.name
        else:
            return False, f"{image_path.name}: Server returned {response.status_code}"
    except Exception as e:
        return False, f"{image_path.name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Batch process images through the OCR server.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--output-dir", type=str, default="app/output", help="Path to save output JSONs.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel requests to send.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images recursively (jpg, jpeg, png)
    extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    # rglob('*') finds all files in all subdirectories
    image_files = [f for f in input_path.rglob('*') if f.suffix in extensions]

    if not image_files:
        print(f"No images found in {input_path} or its subdirectories.")
        return

    print(f"Found {len(image_files)} images. Processing with {args.workers} workers...")

    # Use ThreadPoolExecutor for parallel HTTP requests
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Pass root_input_dir to handle unique filenames
        futures = [executor.submit(process_single_image, img, output_path, input_path) for img in image_files]

        # Wrap with tqdm for a progress bar
        for future in tqdm(futures, desc="OCR Processing"):
            results.append(future.result())

    # Summary
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {len(successes)} / {len(image_files)}")
    if failures:
        print(f"Failures: {len(failures)}")
        with open("batch_failures.log", "w") as log_f:
            for _, err in failures:
                log_f.write(f"{err}\n")
        print("Detailed failures written to batch_failures.log")

if __name__ == "__main__":
    main()
