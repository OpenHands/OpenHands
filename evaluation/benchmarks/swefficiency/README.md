# SWE-fficiency Evaluation

This folder contains the OpenHands inference generation of the [SWE-fficiency benchmark](https://swefficiency.com/) ([paper](https://arxiv.org/pdf/2507.12415v1)).

The evaluation consists of three steps:

1. Environment setup: [install python environment](../../README.md#development-environment) and [configure LLM config](../../README.md#configure-openhands-and-your-llm).
2. [Run inference](#running-inference-locally-with-docker): Generate a edit patch for each Github issue
3. [Evaluate patches](#evaluate-generated-patches)

## Setup Environment and LLM Configuration

Please follow instruction [here](../../README.md#setup) to setup your local development environment and LLM.

## Running inference Locally with Docker

Make sure your Docker daemon is running, and you have ample disk space (at least 200-500GB, depends on the SWE-PErf set you are running on) for the instance-level docker image.

When the `run_infer.sh` script is started, it will automatically pull the relevant SWE-Perf images.
For example, for instance ID `scikit-learn_scikit-learn-11674`, it will try to pull our pre-build docker image `betty1202/sweb.eval.x86_64.scikit-learn_s_scikit-learn-11674` from DockerHub.
This image will be used create an OpenHands runtime image where the agent will operate on.

```bash
./evaluation/benchmarks/swefficiency/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split] [n_runs] [mode]

# Example
./evaluation/benchmarks/swefficiency/scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 500 100 1 swefficiency/swefficiency test
```

where `model_config` is mandatory, and the rest are optional.

- `model_config`, e.g. `eval_gpt4_1106_preview`, is the config group name for your
LLM settings, as defined in your `config.toml`.
- `git-version`, e.g. `HEAD`, is the git commit hash of the OpenHands version you would
like to evaluate. It could also be a release tag like `0.6.2`.
- `agent`, e.g. `CodeActAgent`, is the name of the agent for benchmarks, defaulting
to `CodeActAgent`.
- `eval_limit`, e.g. `10`, limits the evaluation to the first `eval_limit` instances. By
default, the script evaluates the entire SWE-Perf test set (140 issues). Note:
in order to use `eval_limit`, you must also set `agent`.
- `max_iter`, e.g. `20`, is the maximum number of iterations for the agent to run. By
default, it is set to 100.
- `num_workers`, e.g. `3`, is the number of parallel workers to run the evaluation. By
default, it is set to 1.
- `dataset`, a huggingface dataset name. e.g. `SWE-Perf/SWE-Perf`, specifies which dataset to evaluate on.
- `dataset_split`, split for the huggingface dataset. e.g., `test`, `dev`. Default to `test`.

- `n_runs`, e.g. `3`, is the number of times to run the evaluation. Default is 1.
- `mode`, e.g. `swt`, `swt-ci`, or `swe`, specifies the evaluation mode. Default is `swe`.

> [!CAUTION]
> Setting `num_workers` larger than 1 is not officially tested, YMMV.


Let's say you'd like to run 10 instances using `llm.eval_gpt4_1106_preview` and CodeActAgent,

then your command would be:

```bash
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 10
```

### 2. Run the SWE-fficiency benchmark official evaluation

Once the output is converted, use the [official SWE-fficiency benchmark evaluation](https://github.com/swefficiency/swefficiency) to evaluate it.
