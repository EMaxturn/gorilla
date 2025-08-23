import json
import pandas as pd
from pathlib import Path
from llm_judge import llm_judge
import argparse

def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get(rec, *keys, default=""):
    """Return first present key from rec, else default."""
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default

# -------------------------------
# ACCURACY PER QUESTION PER MODEL
# -------------------------------
def evaluate_accuracy_per_question(records, model_hint=None):
    """
    One row per entry with verdicts list and entry_accuracy.
    Compatible with both old & new dataset field names.
    """
    rows = []
    for rec in records:
        q  = _get(rec, "question", "query")
        gt = _get(rec, "ground_truth", "answer")
        model = _get(rec, "model", default=model_hint or "")
        responses = rec.get("model_responses", []) or []

        verdicts = [llm_judge(q, (resp or {}).get("final_answer", ""), gt) for resp in responses]
        success_flags = [v == "success" for v in verdicts]
        entry_accuracy = (sum(success_flags) / len(success_flags)) if success_flags else 0.0

        rows.append({
            "model": model,
            "question": q,
            "ground_truth": gt,
            "verdicts": verdicts,
            "entry_accuracy": entry_accuracy
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------
# RESULT PER QUESTION PER RUN PER MODEL (with traces!)
# ------------------------------------------------------
def evaluate_runs(records, model_hint=None):
    """
    One row per run per entry per model with the requested columns:
    model, run_idx, question, ground_truth, final_answer, verdict, reasoning_trace, human_trace
    """
    rows = []
    for rec in records:
        q  = _get(rec, "question", "query")
        gt = _get(rec, "ground_truth", "answer")
        ht = _get(rec, "human_trace")  # may be empty on old data
        model = _get(rec, "model", default=model_hint or "")
        responses = rec.get("model_responses", []) or []

        for run_idx, resp in enumerate(responses, start=1):
            resp = resp or {}
            final_answer = resp.get("final_answer", "")
            reasoning_trace = resp.get("reasoning_trace", "")

            verdict = llm_judge(q, final_answer, gt)

            rows.append({
                "model": model,
                "run_idx": run_idx,
                "question": q,
                "ground_truth": gt,
                "final_answer": final_answer,
                "verdict": verdict,
                "reasoning_trace": reasoning_trace,
                "human_trace": ht
            })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs with llm_judge.")
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--out-prefix", type=str, default="eval")
    args = parser.parse_args()

    rec_claude = load_records(args.claude)
    rec_gpt    = load_records(args.gpt)
    rec_gemini = load_records(args.gemini)

    df_acc_claude = evaluate_accuracy_per_question(rec_claude, model_hint="claude")
    df_acc_gpt    = evaluate_accuracy_per_question(rec_gpt,    model_hint="gpt")
    df_acc_gemini = evaluate_accuracy_per_question(rec_gemini, model_hint="gemini")

    df_acc_all = pd.concat([df_acc_claude, df_acc_gpt, df_acc_gemini], ignore_index=True)

    acc_per_question_path = f"{args.out_prefix}_accuracy_per_question.csv"
    df_acc_all.to_csv(acc_per_question_path, index=False)

    acc = (
        df_acc_all.groupby("model")["entry_accuracy"]
                  .mean()
                  .reset_index()
                  .rename(columns={"entry_accuracy": "avg_correctness"})
    )
    per_model_path = f"{args.out_prefix}_per_model.csv"
    acc.to_csv(per_model_path, index=False)

    df_runs = pd.concat([
        evaluate_runs(rec_claude, model_hint="claude"),
        evaluate_runs(rec_gpt,    model_hint="gpt"),
        evaluate_runs(rec_gemini, model_hint="gemini"),
    ], ignore_index=True)
    runs_path = f"{args.out_prefix}_runs.csv"
    df_runs.to_csv(runs_path, index=False)

    print("\nFiles written:")
    print(" Per-question accuracy :", acc_per_question_path)
    print(" Per-run details       :", runs_path)
    print(" Per-model summary     :", per_model_path)
    print("\nPer-model average correctness:\n")
    print(acc.to_string(index=False))

if __name__ == "__main__":
    main()
