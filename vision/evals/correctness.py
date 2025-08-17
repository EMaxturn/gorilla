import json
import pandas as pd
from pathlib import Path
from llm_judge import llm_judge   # <- use your existing judge

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

# Paths to your inference files
claude_path = Path("../scripts/inference_outputs/inference_data_003/claude_inference_data_003.json")
gpt_path = Path("../scripts/inference_outputs/inference_data_003/gpt_inference_data_003.json")

df_claude = evaluate_records(load_records(claude_path))
df_gpt = evaluate_records(load_records(gpt_path))

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
