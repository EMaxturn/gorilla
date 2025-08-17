import sys
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repo root to sys.path no matter where we run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vision.handlers.claude_inference_single import run_claude_inference
from vision.handlers.gpt_inference_single import run_gpt_inference


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "inference_outputs")
RUN_DIR_PREFIX = "inference_data_"  # folder prefix, e.g. inference_data_003


def get_next_run_number():
    """
    Find the next available run number by inspecting existing run directories
    under OUTPUT_DIR that match inference_data_XXX.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    existing = [
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith(RUN_DIR_PREFIX)
    ]

    run_numbers = []
    for d in existing:
        m = re.fullmatch(rf"{RUN_DIR_PREFIX}(\d+)", d)
        if m:
            run_numbers.append(int(m.group(1)))

    next_number = max(run_numbers, default=0) + 1
    return f"{next_number:03d}"  # zero-padded to 3 digits


def run_both_models_concurrently(full_img_path, query):
    """
    Launch Claude and GPT inferences concurrently and return their results.
    Any exception from either call is caught and returned as a string so the run continues.
    """
    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(safe_call, run_claude_inference, full_img_path, query): "claude",
            executor.submit(safe_call, run_gpt_inference, full_img_path, query): "gpt",
        }
        results = {"claude": None, "gpt": None}
        for fut in as_completed(futures):
            model = futures[fut]
            results[model] = fut.result()
    return results["claude"], results["gpt"]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/data.json")
        sys.exit(1)

    dataset_path = sys.argv[1]

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_dir = os.path.dirname(dataset_path)

    claude_results = []
    gpt_results = []

    for i, entry in enumerate(dataset, start=1):
        img_path = entry["image"]["path"]
        full_img_path = os.path.join(dataset_dir, "..", img_path)

        query = entry["query"]
        ground_truth = entry.get("answer")

        # ---- Run Claude & GPT concurrently ----
        claude_pred, gpt_pred = run_both_models_concurrently(full_img_path, query)

        print(f"[{i}] Query: {query}")
        print(f"  Claude: {claude_pred}")
        print(f"  GPT   : {gpt_pred}")
        if ground_truth is not None:
            print(f"  Ground Truth: {ground_truth}")
        print()

        base_record = {
            "question": query,
            "ground_truth": ground_truth,
            "image_path": img_path,  # keep for traceability
        }
        claude_results.append({
            **base_record,
            "model": "claude",
            "model_response": claude_pred,
        })
        gpt_results.append({
            **base_record,
            "model": "gpt",
            "model_response": gpt_pred,
        })

    # -------- Save results into a per-run folder --------
    run_number = get_next_run_number()
    run_dir = os.path.join(OUTPUT_DIR, f"{RUN_DIR_PREFIX}{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    claude_out = os.path.join(run_dir, f"claude_inference_data_{run_number}.json")
    gpt_out = os.path.join(run_dir, f"gpt_inference_data_{run_number}.json")

    with open(claude_out, "w", encoding="utf-8") as f:
        json.dump(claude_results, f, indent=2, ensure_ascii=False)

    with open(gpt_out, "w", encoding="utf-8") as f:
        json.dump(gpt_results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(claude_results)} Claude results to {claude_out}")
    print(f"Saved {len(gpt_results)} GPT results to {gpt_out}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
