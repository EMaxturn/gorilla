import sys
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- Configuration ---
# Define models and their corresponding inference functions
MODELS_TO_RUN = {
    "claude": "vision.handlers.claude_inference_single.run_claude_inference",
    "gpt": "vision.handlers.gpt_inference_single.run_gpt_inference",
    "gemini": "vision.handlers.gemini_inference_single.run_gemini_inference",
}
N_RUNS = 3
OUTPUT_DIR_NAME = "inference_outputs"
RUN_DIR_PREFIX = "inference_data_"

# --- make stdout line-buffered & flush by default ---
try:
    sys.stdout.reconfigure(line_buffering=True)
except TypeError:  # For environments where reconfigure is not available
    pass
PRINT = lambda *a, **k: print(*a, **{**k, "flush": True})


# --- Dynamically import inference functions ---
def import_from_string(path: str):
    """Imports a function from a string path like 'module.submodule.function'."""
    module_path, function_name = path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


try:
    INFERENCE_FUNCTIONS = {
        name: import_from_string(path) for name, path in MODELS_TO_RUN.items()
    }
except (ImportError, AttributeError) as e:
    PRINT(f"[ERROR] Could not import an inference function: {e}")
    sys.exit(1)


# --- formats each model result into the target schema ---
def format_result(res: Any) -> Dict[str, str]:
    if isinstance(res, dict):
        return {
            "final_answer": str(res.get("final_answer", "")),
            "reasoning_trace": str(res.get("reasoning_trace", "")),
        }
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        return {"final_answer": str(res[0]), "reasoning_trace": str(res[1])}
    # Fallback (e.g., error string)
    return {"final_answer": str(res), "reasoning_trace": ""}


# --- Core Logic ---
def get_next_run_dir(base_output_dir: Path) -> Path:
    """Calculates the next sequential run directory path."""
    base_output_dir.mkdir(exist_ok=True)
    existing_runs = [
        int(d.name.replace(RUN_DIR_PREFIX, ""))
        for d in base_output_dir.iterdir()
        if d.is_dir() and d.name.startswith(RUN_DIR_PREFIX) and d.name.replace(RUN_DIR_PREFIX, "").isdigit()
    ]
    next_run_num = max(existing_runs, default=0) + 1
    run_dir = base_output_dir / f"{RUN_DIR_PREFIX}{next_run_num:03d}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def run_models_multiple(
        full_img_path: Path, query: str, ground_truth: str, n_runs: int = N_RUNS
) -> Tuple[List[str], List[str], List[str]]:
    """
    Runs inference for all models for N runs.
    The runs are sequential, but the models within each run are called in parallel.
    """
    # Initialize results structure
    results: Dict[str, List[Any]] = {name: [None] * n_runs for name in INFERENCE_FUNCTIONS}

    def safe_call(fn, *args, **kwargs):
        """Wrapper to catch exceptions during threaded execution."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"[ERROR] in {fn.__name__}: {type(e).__name__}: {e}"

    # This outer loop is SEQUENTIAL
    for n in range(n_runs):
        PRINT(f"[Run {n + 1}/{n_runs}] Starting parallel inference for all models...")

        # A new thread pool is created for each sequential run.
        # This ensures that Run (n) completes entirely before Run (n+1) begins.
        with ThreadPoolExecutor(max_workers=len(INFERENCE_FUNCTIONS)) as executor:
            # Submitting all model calls to the thread pool to run in PARALLEL
            future_to_model = {
                executor.submit(safe_call, func, str(full_img_path), query): name
                for name, func in INFERENCE_FUNCTIONS.items()
            }

            # Gather results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                result = future.result()
                # --- format to {"final_answer": ..., "reasoning_trace": ...}
                results[model_name][n] = format_result(result)

        # Print results for the completed run
        PRINT(f"--- Results for Run {n + 1} ---")
        PRINT(f"Query : {query}")
        for name, res_list in results.items():
            PRINT(f"{name.capitalize():<7}: {res_list[n]}")
        if ground_truth is not None:
            PRINT(f"Ground Truth: {ground_truth}")
        PRINT("-" * (23 + len(str(n + 1))))
        PRINT()

    return tuple(results[name] for name in MODELS_TO_RUN.keys())


def main():
    """Main script execution."""
    PRINT("[Boot] Starting inference script…")

    if len(sys.argv) != 2:
        PRINT(f"Usage: {Path(sys.argv[0]).name} path/to/data.json")
        sys.exit(1)

    dataset_path = Path(sys.argv[1]).resolve()
    if not dataset_path.is_file():
        PRINT(f"[ERROR] Dataset file not found at: {dataset_path}")
        sys.exit(1)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_dir = dataset_path.parent

    # Store results for all entries
    all_results: Dict[str, List[Dict]] = {name: [] for name in INFERENCE_FUNCTIONS}

    for idx, entry in enumerate(dataset, start=1):
        PRINT(f"=========== Starting Batch {idx}/{len(dataset)} ===========")

        img_path = Path(entry["image"]["path"])
        # Assuming image path in JSON is relative to the project root
        full_img_path = (dataset_dir.parent / img_path).resolve()
        query = entry["query"]
        ground_truth = entry.get("answer")

        # Run the core logic for the current dataset entry
        model_predictions = run_models_multiple(full_img_path, query, ground_truth)

        # Unpack predictions and format results
        claude_preds, gpt_preds, gemini_preds = model_predictions

        base_record = {"question": query, "ground_truth": ground_truth, "image_path": str(img_path)}

        all_results["claude"].append({**base_record, "model": "claude", "model_responses": claude_preds})
        all_results["gpt"].append({**base_record, "model": "gpt", "model_responses": gpt_preds})
        all_results["gemini"].append({**base_record, "model": "gemini", "model_responses": gemini_preds})

        PRINT(f"=========== Finished Batch {idx}/{len(dataset)} ===========\n")

    # Determine output directory and save results
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR_NAME
    run_dir = get_next_run_dir(output_dir)
    run_number = run_dir.name.replace(RUN_DIR_PREFIX, "")

    for model_name, results_list in all_results.items():
        output_file = run_dir / f"{model_name}_inference_data_{run_number}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        PRINT(f"Saved {len(results_list)} results for {model_name.capitalize()} to {output_file}")

    PRINT(f"\n[Done] All inferences complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()