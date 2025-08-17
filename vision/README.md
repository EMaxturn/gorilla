# Vision Tool-Use Benchmark

### _\*\*All of this is subject to change_

## Adding to the datset

- Add your image to the correct category in `vision/images`, if there is no existing category matching your image create a new one.
- Create an entry for your image, query, ground truth, category, etc. in `vision/dataset/dataset_v1.json`

## Running an iteration of inference

Run the following command from the root of the repo:
`python3 vision/scripts/run_all_inference.py vision/dataset/dataset_v1.json`

## Running evaluation on an iteration of inference

Run the following command from the root of the repo. Replace XXX with the iteration you want to evaluate:
`python3 vision/evals/correctness.py    --claude vision/scripts/inference_outpu
ts/inference_data_XXX/claude_inference_data_XXX.json   --gpt vision/scripts/inference_outputs/inference_data_XXX/gpt_
inference_data_XXX.json`
