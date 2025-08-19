import sys
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- make stdout line-buffered & flush by default ---
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
PRINT = lambda *a, **k: print(*a, **{**k, "flush": True})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vision.handlers.claude_inference_single import run_claude_inference
from vision.handlers.gpt_inference_single import run_gpt_inference
from vision.handlers.gemini_inference_single import run_gemini_inference


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "inference_outputs")
RUN_DIR_PREFIX = "inference_data_"
N_RUNS = 3


def get_next_run_number():
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
    return f"{next_number:03d}"


def run_models_concurrently(full_img_path, query):
    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"

    # 3 workers since we launch 3 tasks
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(safe_call, run_claude_inference, full_img_path, query): "claude",
            executor.submit(safe_call, run_gpt_inference, full_img_path, query): "gpt",
            executor.submit(safe_call, run_gemini_inference, full_img_path, query): "gemini",
        }
        results = {"claude": None, "gpt": None, "gemini": None}
        for fut in as_completed(futures):
            model = futures[fut]
            results[model] = fut.result()
    return results["claude"], results["gpt"], results["gemini"]


def run_models_multiple(full_img_path, query, ground_truth, n_runs=N_RUNS):
    results = {"claude": [], "gpt": [], "gemini": []}
    for n in range(n_runs):
        claude_pred, gpt_pred, gemini_pred = run_models_concurrently(full_img_path, query)
        results["claude"].append(claude_pred)
        results["gpt"].append(gpt_pred)
        results["gemini"].append(gemini_pred)

        PRINT(f"[Run {n+1}] Query: {query}")
        PRINT(f"Claude: {claude_pred}")
        PRINT(f"GPT   : {gpt_pred}")
        PRINT(f"Gemini: {gemini_pred}")
        if ground_truth is not None:
            PRINT(f"Ground Truth: {ground_truth}")
        PRINT()
    return results["claude"], results["gpt"], results["gemini"]


def main():
    PRINT("[Boot] starting inference script…")

    if len(sys.argv) != 2:
        PRINT(f"Usage: {sys.argv[0]} path/to/data.json")
        sys.exit(1)

    dataset_path = sys.argv[1]
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_dir = os.path.dirname(dataset_path)
    claude_results, gpt_results, gemini_results = [], [], []

    for idx, entry in enumerate(dataset, start=1):
        PRINT(f"[Starting New Batch] Batch {idx}/{len(dataset)}")

        img_path = entry["image"]["path"]
        full_img_path = os.path.join(dataset_dir, "..", img_path)
        query = entry["query"]
        ground_truth = entry.get("answer")

        claude_preds, gpt_preds, gemini_preds = run_models_multiple(full_img_path, query, ground_truth)
        base_record = {"question": query, "ground_truth": ground_truth, "image_path": img_path}
        claude_results.append({**base_record, "model": "claude", "model_responses": claude_preds})
        gpt_results.append({**base_record, "model": "gpt", "model_responses": gpt_preds})
        gemini_results.append({**base_record, "model": "gemini", "model_responses": gemini_preds})

    run_number = get_next_run_number()
    run_dir = os.path.join(OUTPUT_DIR, f"{RUN_DIR_PREFIX}{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, f"claude_inference_data_{run_number}.json"), "w", encoding="utf-8") as f:
        json.dump(claude_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, f"gpt_inference_data_{run_number}.json"), "w", encoding="utf-8") as f:
        json.dump(gpt_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, f"gemini_inference_data_{run_number}.json"), "w", encoding="utf-8") as f:
        json.dump(gemini_results, f, indent=2, ensure_ascii=False)

    PRINT(f"Saved {len(claude_results)} Claude results")
    PRINT(f"Saved {len(gpt_results)} GPT results")
    PRINT(f"Saved {len(gemini_results)} Gemini results")
    PRINT(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
