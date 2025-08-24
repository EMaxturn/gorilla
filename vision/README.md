# Vision Tool-Use Benchmark

### _\*\*All of this is subject to change_

## Install all dependencies
- Simply run `python3 -m pip install -r vision/requirements.txt`

## Adding to the datset

- Add your image to the correct category in `vision/images`, if there is no existing category matching your image create a new one.
- Create an entry for your image, query, ground truth, category, etc. in `vision/dataset/dataset_v1.json`

## Running inference batches

At the top of `run_all_inference.py` set `N_RUNS` to the number of times you want to run the same query per model.

Run the following command from the root of the repo:
`python3 -m vision.scripts.run_all_inference vision/dataset/dataset_v1.json`

You will find the result in the latest entry inside of `vision/scripts/inference_outputs/`

## Running evaluation on an iteration of inference

Run the following command from the root of the repo. Replace XXX with the iteration you want to evaluate:

```
python3 vision/evals/evaluate_correctness.py \
  --claude vision/scripts/inference_outputs/inference_data_004/claude_inference_data_004.json \
  --gpt    vision/scripts/inference_outputs/inference_data_004/gpt_inference_data_004.json \
  --gemini vision/scripts/inference_outputs/inference_data_004/gemini_inference_data_004.json \
  --out-prefix vision/scripts/inference_outputs/inference_data_004/eval_run004
```

You will find the resulting csv files in the folder you ran evals on inside `vision/scripts/inference_outputs/inference_data_XXX`
