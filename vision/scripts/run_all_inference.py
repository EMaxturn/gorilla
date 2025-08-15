import sys
import os
import json
import re

# Add repo root to sys.path no matter where we run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vision.handlers.claude_inference_single import run_claude_inference


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "inference_outputs")


def get_next_run_number(prefix="inference_data_", ext=".json"):
    """Find the next available run number for the inference file in OUTPUT_DIR."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix) and f.endswith(ext)]
    
    run_numbers = []
    for f in existing_files:
        match = re.search(rf"{prefix}(\d+){ext}", f)
        if match:
            run_numbers.append(int(match.group(1)))

    next_number = max(run_numbers, default=0) + 1
    return f"{next_number:03d}"  # zero-padded to 3 digits


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/data.json")
        sys.exit(1)

    dataset_path = sys.argv[1]

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_dir = os.path.dirname(dataset_path)

    results = []
    for i, entry in enumerate(dataset, start=1):
        img_path = entry["image"]["path"]
        full_img_path = os.path.join(dataset_dir, "..", img_path)

        query = entry["query"]
        ground_truth = entry.get("answer")

        pred = run_claude_inference(full_img_path, query)

        print(f"[{i}] Query: {query}")
        print(f"  Model: {pred}")
        if ground_truth is not None:
            print(f"  Ground Truth : {ground_truth}")
        print()

        results.append({
            "question": query,
            "model_response": pred,
            "ground_truth": ground_truth,
            "image_path": img_path  # keep for traceability
        })

    # Save results into numbered file in OUTPUT_DIR
    run_number = get_next_run_number()
    out_filename = os.path.join(OUTPUT_DIR, f"inference_data_{run_number}.json")
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {out_filename}")


if __name__ == "__main__":
    main()
