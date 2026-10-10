import argparse
import requests
import json
import time
from pathlib import Path
from client.config import SERVER_URL, RETRY_SETTINGS
from client.postprocess import normalize_text, detect_category

def main():
    """
    Main entry point for the OCR client.
    Reads an image, requests a prediction from the local daemon server,
    normalizes the result based on detected category, and saves the output to JSON.
    """
    parser = argparse.ArgumentParser(description="AMD OCR Challenge Client")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input file {args.input_image} not found.")
        return

    # Define output path as per requirements: /app/output/<image_name>_output.json
    image_name = input_path.stem
    output_dir = Path("/app/output")
    # For local dev, if /app/output doesn't exist, we'll use a local output folder
    if not output_dir.exists():
        output_dir = Path("app/output")
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{image_name}_output.json"

    try:
        # 1. Request prediction from the local server with retries
        max_retries = RETRY_SETTINGS["max_retries"]
        retry_delay = RETRY_SETTINGS["retry_delay"]
        result = None

        for attempt in range(max_retries):
            try:
                with open(input_path, "rb") as f:
                    files = {"file": (input_path.name, f)}
                    response = requests.post(SERVER_URL, files=files, timeout=25)
                    response.raise_for_status()
                    result = response.json()
                    break
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"All {max_retries} attempts failed: {e}")

        if result is None:
            # Fail-safe: create a valid output file even if the server is down
            raw_text = ""
            confidence = 0.0
        else:
            raw_text = result.get("text", "")
            confidence = result.get("confidence", 0.0)

        # 2. Determine category and normalize text
        category = detect_category(raw_text)
        normalized_text = normalize_text(raw_text, category)

        # 3. Save output JSON
        output_data = {
            "text": normalized_text,
            "confidence": confidence
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Successfully processed {input_path.name} -> {output_file}")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
