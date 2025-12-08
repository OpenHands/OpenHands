# TestGenEval Benchmark Evaluation

This folder contains the evaluation harness for the TestGenEval benchmark, which is based on the original TestGenEval benchmark ([paper](https://arxiv.org/abs/2410.00752)). TestGenEval is designed to evaluate the ability of language models to generate unit tests for given Python functions.

## Setup Environment and LLM Configuration

1. Follow the instructions [here](../../README.md#setup) to set up your local development environment and configure your LLM.

2. Install the TestGenEval dependencies:
```bash
poetry install --with testgeneval
```

## Run Inference

To generate tests using your model, run the following command:

```bash
./evaluation/benchmarks/testgeneval/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split]

# Example
./evaluation/benchmarks/testgeneval/scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 100 30 1 kjain14/testgenevallite test
```

Parameters:
- `model_config`: The config group name for your LLM settings (e.g., `eval_gpt4_1106_preview`)
- `git-version`: The git commit hash or release tag of OpenHands to evaluate (e.g., `HEAD` or `0.6.2`)
- `agent`: The name of the agent for benchmarks (default: `CodeActAgent`)
- `eval_limit`: Limit the evaluation to the first N instances (optional)
- `max_iter`: Maximum number of iterations for the agent to run (default: 30)
- `num_workers`: Number of parallel workers for evaluation (default: 1)
- `dataset`: HuggingFace dataset name (default: `kjain14/testgenevallite`)
- `dataset_split`: Dataset split to use (default: `test`)

After running the inference, you will obtain an `output.jsonl` file (by default saved to `evaluation/evaluation_outputs`).

## Evaluate Generated Tests

To evaluate the generated tests, use the `eval_infer.sh` script:

```bash
./evaluation/benchmarks/testgeneval/scripts/eval_infer.sh $YOUR_OUTPUT_JSONL [instance_id] [dataset_name] [split] [num_workers] [skip_mutation]

# Example
./evaluation/benchmarks/testgeneval/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/kjain14__testgenevallite-test/CodeActAgent/gpt-4-1106-preview_maxiter_50_N_v1.0/output.jsonl
```

Optional arguments:
- `instance_id`: Evaluate a single instance (optional)
- `dataset_name`: Name of the dataset to use (default: `kjain14/testgenevallite`)
- `split`: Dataset split to use (default: `test`)
- `num_workers`: Number of workers for running docker (default: 1)
- `skip_mutation`: Skip mutation testing (enter `true` if desired)

The evaluation results will be saved to `evaluation/evaluation_outputs/outputs/kjain14__testgenevallite-test/CodeActAgent/gpt-4-1106-preview_maxiter_50_N_v1.0/` with `output.testgeneval.jsonl` containing the metrics.

## Metrics

The TestGenEval benchmark evaluates generated tests based on the following metrics:

1. Correctness: Measures if the generated tests are syntactically correct and run without errors.
2. Coverage: Assesses the code coverage achieved by the generated tests.
3. Mutation Score: Evaluates the effectiveness of the tests in detecting intentionally introduced bugs (mutations).
4. Readability: Analyzes the readability of the generated tests using various metrics.

## Submit Your Evaluation Results

To contribute your evaluation results:

1. Fork [our HuggingFace evaluation outputs](https://huggingface.co/spaces/OpenHands/evaluation).
2. Add your results to the forked repository.
3. Submit a Pull Request with your evaluation results following the guide [here](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions#pull-requests-and-discussions).

## Additional Resources

- [TestGenEval Paper](https://arxiv.org/abs/2410.00752)
- [OpenHands Documentation](https://github.com/OpenHands/OpenHands)
- [HuggingFace Datasets](https://huggingface.co/datasets)

For any questions or issues, please open an issue in the [OpenHands repository](https://github.com/OpenHands/OpenHands/issues).
