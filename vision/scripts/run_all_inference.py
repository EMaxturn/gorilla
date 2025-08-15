import sys, os,json

# Add repo root to sys.path no matter where we run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vision.handlers.claude_inference_single import run_claude_inference

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/data.json")
        sys.exit(1)

    dataset_path = sys.argv[1]

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_dir = os.path.dirname(dataset_path)

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

if __name__ == "__main__":
    main()
