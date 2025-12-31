# Tau-Bench Evaluation

This directory contains the evaluation scripts for Tau-Bench.

## Setup

First, make sure you have installed the `tau-bench` package:

```bash
pip install tau-bench
```

## Running Evaluation

To run the evaluation, use the following command:

```bash
python evaluation/benchmarks/tau_bench/run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config <your_llm_config> \
  --env retail
```
