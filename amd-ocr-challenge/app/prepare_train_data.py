import csv
import json
import os
from typing import List, Dict

def prepare_qwen_vl_dataset(csv_path: str, dataset_root: str, output_dir: str):
    """
    Converts the project's plates.csv into Qwen-2.5 VL conversation format JSONL.
    """
    print(f"Processing dataset from {csv_path}...")

    splits = {"train": [], "valid": []}

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV columns: 'class id', 'filepaths', 'labels', 'data set'
            split = row['data set'].strip()
            if split not in splits:
                continue

            image_rel_path = row['filepaths'].strip()
            label = row['labels'].strip()

            # Absolute path for the VLM processor
            abs_image_path = os.path.abspath(os.path.join(dataset_root, image_rel_path))

            # Qwen-2.5 VL Conversation Template
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": abs_image_path},
                            {"type": "text", "text": "Extract all text from this image. Output only the extracted text."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": label}
                        ]
                    }
                ]
            }
            splits[split].append(conversation)

    # Save to JSONL
    os.makedirs(output_dir, exist_ok=True)
    for split_name, data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}_data.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Saved {len(data)} entries to {output_file}")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "amd-ocr-challenge/app/dataset/plates.csv"
    DATASET_ROOT = "amd-ocr-challenge/app/dataset"
    OUTPUT_DIR = "amd-ocr-challenge/app/dataset/qwen_formatted"

    prepare_qwen_vl_dataset(CSV_PATH, DATASET_ROOT, OUTPUT_DIR)
