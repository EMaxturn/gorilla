import json
import pandas as pd
from pathlib import Path
from llm_judge import llm_judge
import argparse

def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def eval_record(record):
    q = record.get("question", "")
    gt = record.get("ground_truth", "")
    model = record.get("model", "")
    responses = record.get("model_responses", [])

    verdicts = [llm_judge(q, resp, gt) for resp in responses]
    success_flags = [v == "success" for v in verdicts]

    entry_accuracy = (sum(success_flags) / len(success_flags)) if success_flags else 0.0

    return {
        "model": model,
        "question": q,
        "ground_truth": gt,
        "verdicts": verdicts,             # e.g. ["success","fail","success"]
        "entry_accuracy": entry_accuracy  # 0.0–1.0
    }

def evaluate_file(records):
    return pd.DataFrame([eval_record(rec) for rec in records])

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-run model outputs with llm_judge.")
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--out-prefix", type=str, default="eval")
    args = parser.parse_args()

    df_claude = evaluate_file(load_records(args.claude))
    df_gpt    = evaluate_file(load_records(args.gpt))
    df_gemini = evaluate_file(load_records(args.gemini))

    df_all = pd.concat([df_claude, df_gpt, df_gemini], ignore_index=True)

    # Average correctness per model (just mean of entry accuracies)
    acc = (
        df_all.groupby("model")["entry_accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"entry_accuracy": "avg_correctness"})
    )

    # Save
    df_all.to_csv(f"{args.out_prefix}_entries.csv", index=False)
    acc.to_csv(f"{args.out_prefix}_per_model.csv", index=False)

    print("\nPer entry correctness written to:", f"{args.out_prefix}_entries.csv")
    print("Per model average correctness:\n")
    print(acc.to_string(index=False))

if __name__ == "__main__":
    main()
