# Multi-swe-bench Evaluation with OpenHands

## LLM Setup

Please follow [here](../../README.md#setup).

## Dataset Preparing

Please download the [**Multi-SWE-Bench** dataset](https://huggingface.co/datasets/bytedance-research/Multi-SWE-Bench).
And change the dataset following [script](scripts/data/data_change.py).

```bash
python evaluation/benchmarks/multi_swe_bench/scripts/data/data_change.py
```

## Docker image download

Please download the multi-swe-bench docker images from [here](https://github.com/multi-swe-bench/multi-swe-bench?tab=readme-ov-file#run-evaluation).

## Generate patch

Please edit the [script](infer.sh) and run it.

```bash
bash evaluation/benchmarks/multi_swe_bench/infer.sh
```

Script variable explanation:

- `models`, e.g. `llm.eval_gpt4_1106_preview`, is the config group name for your
LLM settings, as defined in your `config.toml`.
- `git-version`, e.g. `HEAD`, is the git commit hash of the OpenHands version you would
like to evaluate. It could also be a release tag like `0.6.2`.
- `agent`, e.g. `CodeActAgent`, is the name of the agent for benchmarks, defaulting to `CodeActAgent`.
- `eval_limit`, e.g. `10`, limits the evaluation to the first `eval_limit` instances. By
default, the script evaluates the (500 issues), which will no exceed the maximum of the dataset number.
- `max_iter`, e.g. `20`, is the maximum number of iterations for the agent to run. By
default, it is set to 50.
- `num_workers`, e.g. `3`, is the number of parallel workers to run the evaluation. By
default, it is set to 1.
- `language`, the language of your evaluating dataset.
- `dataset`, the absolute position of the dataset jsonl.

**Skipping errors on build**

For debugging purposes, you can set `export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true` to continue evaluation even when instances reach maximum retries. After evaluation completes, check `maximum_retries_exceeded.jsonl` for a list of failed instances, fix those issues, and then run the evaluation again with `export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=false`.

The results will be generated in evaluation/evaluation_outputs/outputs/XXX/CodeActAgent/YYY/output.jsonl, you can refer to the [example](examples/output.jsonl).

## Running evaluation

First, install [multi-swe-bench](https://github.com/multi-swe-bench/multi-swe-bench).

```bash
pip install multi-swe-bench
```

Second, convert the output.jsonl to patch.jsonl with [script](scripts/eval/convert.py), you can refer to the [example](examples/patch.jsonl).

```bash
python evaluation/benchmarks/multi_swe_bench/scripts/eval/convert.py
```

Finally, evaluate with multi-swe-bench.
The config file config.json can be refer to the [example](examples/config.json) or [github](https://github.com/multi-swe-bench/multi-swe-bench/tree/main?tab=readme-ov-file#configuration-file-example).

```bash
python -m multi_swe_bench.harness.run_evaluation --config config.json
```
