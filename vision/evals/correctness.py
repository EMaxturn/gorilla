import json
import pandas as pd
from pathlib import Path
from llm_judge import llm_judge
import argparse


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_records(rows):
    out = []
    for i, r in enumerate(rows, 1):
        q = r.get("question", "")
        gt = r.get("ground_truth", "")
        resp = r.get("model_response", "")
        verdict = llm_judge(q, resp, gt)
        out.append({
            "idx": i,
            "model": r.get("model", ""),
            "question": q,
            "ground_truth": gt,
            "model_response": resp,
            "verdict": verdict
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth.")
    parser.add_argument("--claude", type=Path, required=True, help="Path to Claude JSON file")
    parser.add_argument("--gpt", type=Path, required=True, help="Path to GPT JSON file")
    args = parser.parse_args()

    df_claude = evaluate_records(load_records(args.claude))
    df_gpt = evaluate_records(load_records(args.gpt))

    df_all = pd.concat([df_claude, df_gpt], ignore_index=True)

    # Accuracy per model
    acc = (
        df_all.assign(correct=df_all["verdict"].eq("success"))
              .groupby("model")["correct"]
              .mean()
              .reset_index()
              .rename(columns={"correct": "accuracy"})
    )

    print(acc)
    df_all.to_csv("model_eval_results.csv", index=False)
    acc.to_csv("model_eval_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
