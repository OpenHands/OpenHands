# BFCL (Berkeley Function-Calling Leaderboard) Evaluation

This directory contains the evaluation scripts for BFCL.

## Setup

You may need to clone the official BFCL repository or install the evaluation package if available.

```bash
# Example setup (adjust as needed)
# git clone https://github.com/ShishirPatil/gorilla.git
# cd gorilla/berkeley-function-call-leaderboard
# pip install -r requirements.txt
```

## Running Evaluation

To run the evaluation, you need to provide the path to the BFCL dataset:

```bash
python evaluation/benchmarks/bfcl/run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config <your_llm_config> \
  --dataset-path /path/to/bfcl_dataset.json
```
