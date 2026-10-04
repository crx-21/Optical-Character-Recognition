import os
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

def load_ground_truth(csv_path):
    """
    Loads the plates.csv ground truth file.
    Columns: [class id, filepaths, labels, data set]
    """
    ground_truth = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                continue

            ground_truth.append({
                'id': row[0],
                'filepath': row[1],
                'label': row[2],
                'dataset': row[3] if len(row) > 3 else 'unknown'
            })
    return ground_truth

def load_prediction(output_dir, image_relative_path):
    """
    Attempts to find the output JSON for a given image relative path.
    """
    unique_name = image_relative_path
    for char in ['/', chr(92), '.']:
        unique_name = unique_name.replace(char, '_')
    
    json_filename = f"{unique_name}_output.json"
    json_path = Path(output_dir) / json_filename

    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def calculate_metrics(gt_list, predictions):
    """
    Calculates accuracy and generates a character confusion matrix.
    """
    correct = 0
    total = len(gt_list)
    confusion_matrix = defaultdict(Counter)
    confidence_scores = []

    for gt, pred in zip(gt_list, predictions):
        label = gt['label'].strip()
        pred_text = pred['text'].strip() if pred else ""
        conf = pred['confidence'] if pred else 0.0

        confidence_scores.append(conf)

        if label == pred_text:
            correct += 1
        else:
            if 0 < abs(len(label) - len(pred_text)) <= 2:
                for i in range(min(len(label), len(pred_text))):
                    if label[i] != pred_text[i]:
                        confusion_matrix[label[i]][pred_text[i]] += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, confusion_matrix, confidence_scores

def main():
    CSV_PATH = "app/dataset/plates.csv"
    OUTPUT_DIR = "app/output"

    if not os.path.exists(CSV_PATH):
        print(f"Error: Ground truth file {CSV_PATH} not found.")
        return

    print(f"Loading ground truth from {CSV_PATH}...")
    gt_list = load_ground_truth(CSV_PATH)
    print(f"Loaded {len(gt_list)} samples.")

    predictions = []
    missing_files = 0

    print(f"Matching predictions from {OUTPUT_DIR}...")
    for gt in gt_list:
        pred = load_prediction(OUTPUT_DIR, gt['filepath'])
        predictions.append(pred)
        if pred is None:
            missing_files += 1

    if missing_files > 0:
        print(f"Warning: {missing_files} predictions were not found in {OUTPUT_DIR}.")

    accuracy, confusion, confs = calculate_metrics(gt_list, predictions)

    print("\n" + "="*30)
    print(" OCR EVALUATION REPORT ")
    print("="*30)
    print(f"Total Samples:    {len(gt_list)}")
    print(f"Exact Match Acc:  {accuracy:.2f}%")
    print(f"Missing Results:  {missing_files}")

    if confs:
        avg_conf = sum(confs) / len(confs)
        print(f"Avg Confidence:   {avg_conf:.4f}")

    print("\nTop Character Confusions:")
    all_confusions = []
    for char, counts in confusion.items():
        for pred_char, count in counts.items():
            all_confusions.append((char, pred_char, count))

    all_confusions.sort(key=lambda x: x[2], reverse=True)
    for char, pred_char, count in all_confusions[:15]:
        print(f"  {char} -> {pred_char}: {count} times")

    print("="*30)

if __name__ == "__main__":
    main()
