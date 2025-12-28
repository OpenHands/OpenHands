#!/bin/bash

# NOTE: this script is for rolling out the Multi-SWE-Gym dataset for **TRAINING**
# For more information, please refer to
# 1. the Github Repo: https://github.com/SWE-Gym/SWE-Gym
# 2. the paper: https://arxiv.org/abs/2412.21139

MODEL=$1  # eg your llm config name in config.toml (eg: "llm.claude-3-5-sonnet-20241022-t05")
EXP_NAME=$2 # "train-t05"
EVAL_DATASET=$3  # path to original dataset (jsonl file)
MAX_ITER=$4
N_WORKERS=${5:-64}
N_RUNS=${6:-1}
EVAL_LIMIT=${7:-}
SKIP_IDS_THRESHOLD=$8
SKIP_IDS_PATTERN=$9
INPUT_SKIP_IDS=${10}
FILTER_DATASET_AFTER_SAMPLING=${11:-}

export EXP_NAME=$EXP_NAME
# use 2x resources for rollout since some codebases are pretty resource-intensive
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
echo "MODEL: $MODEL"
echo "EXP_NAME: $EXP_NAME"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "INPUT_SKIP_IDS: $INPUT_SKIP_IDS"
# Generate DATASET path by adding _with_runtime_ before .jsonl extension
DATASET="${EVAL_DATASET%.jsonl}_with_runtime_.jsonl"  # path to converted dataset

# Create the converted dataset file
echo "Creating converted dataset at: $DATASET"
poetry run python ./evaluation/benchmarks/multi_swe_bench/scripts/data/data_change.py --input "$EVAL_DATASET" --output "$DATASET"

SPLIT="train"
export LANGUAGE=java

if [ -z "$ALLHANDS_API_KEY" ] || [ "$RUNTIME" != "remote" ]; then
    echo "ALLHANDS_API_KEY is not set or RUNTIME is not set to remote. Will rollout and evaluate locally using Docker. WARNING: A large value of N_WORKERS will result in a large number of Docker containers being spun up and may crash your machine."
    export RUNTIME=docker
else
    echo "ALLHANDS_API_KEY is set and RUNTIME is set to remote. Continuing rollout and evaluation with remote runtime..."
    export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
fi


# ===== Run inference =====
source "evaluation/utils/version_control.sh"
get_openhands_version

echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "EVAL_DOCKER_IMAGE_PREFIX: $EVAL_DOCKER_IMAGE_PREFIX"

# Default to NOT use Hint
export USE_INSTANCE_IMAGE=true
export USE_HINT_TEXT=false
export RUN_WITH_BROWSING=false
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION-no-hint-$EXP_NAME"

function run_eval() {
  local eval_note=$1
  export LANGUAGE=java
  echo "About to run command"
  COMMAND="EVAL_DOCKER_IMAGE_PREFIX=$EVAL_DOCKER_IMAGE_PREFIX; LANGUAGE=java;
    poetry run python evaluation/benchmarks/multi_swe_bench/run_infer.py \
    --agent-cls CodeActAgent \
    --llm-config $MODEL \
    --max-iterations $MAX_ITER \
    --eval-num-workers $N_WORKERS \
    --eval-note $eval_note \
    --dataset $DATASET \
    --split $SPLIT"

  # Conditionally add filter flag
  if [ "$FILTER_DATASET_AFTER_SAMPLING" = "true" ]; then
    COMMAND="$COMMAND --filter_dataset_after_sampling"
  fi

  echo "Running command: $COMMAND"
  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  eval $COMMAND
}

for run_idx in $(seq 1 $N_RUNS); do
    if [ -n "$SKIP_IDS_THRESHOLD" ]; then
        echo "Computing SKIP_IDS for run $run_idx..."
        SKIP_CMD="poetry run python evaluation/benchmarks/multi_swe_bench/compute_skip_ids.py $SKIP_IDS_THRESHOLD"
        if [ -n "$SKIP_IDS_PATTERN" ]; then
            SKIP_CMD="$SKIP_CMD --pattern \"$SKIP_IDS_PATTERN\""
        fi
        COMPUTED_SKIP_IDS=$(eval $SKIP_CMD)
        SKIP_STATUS=$?
        if [ $SKIP_STATUS -ne 0 ]; then
            echo "ERROR: Skip IDs computation failed with exit code $SKIP_STATUS"
            exit $SKIP_STATUS
        fi
        echo "COMPUTED_SKIP_IDS: $COMPUTED_SKIP_IDS"
    else
        echo "SKIP_IDS_THRESHOLD not provided, skipping SKIP_IDS computation"
        COMPUTED_SKIP_IDS=""
    fi

    # Concatenate COMPUTED_SKIP_IDS and INPUT_SKIP_IDS
    if [ -n "$COMPUTED_SKIP_IDS" ] && [ -n "$INPUT_SKIP_IDS" ]; then
        export SKIP_IDS="${COMPUTED_SKIP_IDS},${INPUT_SKIP_IDS}"
    elif [ -n "$COMPUTED_SKIP_IDS" ]; then
        export SKIP_IDS="$COMPUTED_SKIP_IDS"
    elif [ -n "$INPUT_SKIP_IDS" ]; then
        export SKIP_IDS="$INPUT_SKIP_IDS"
    else
        unset SKIP_IDS
    fi

    echo "FINAL SKIP_IDS: $SKIP_IDS"
    echo ""

    while true; do
        echo "### Running inference... ###"
        unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
        current_eval_note="$EVAL_NOTE-run_$run_idx"
        echo "EVAL_NOTE: $current_eval_note"
        echo "DATASET command: $DATASET"
        #INFER_OUTPUT=$(run_eval $current_eval_note)
        INFER_OUTPUT=$(run_eval $current_eval_note | tee /dev/stderr)
        INFER_STATUS=$?  # Capture the exit status of run_infer.sh
        echo "INFER_STATUS: $INFER_STATUS"

        echo "### Cleaning up remote runtime... ###"
        ./evaluation/utils/scripts/cleanup_remote_runtime.sh

        if [ $INFER_STATUS -eq 0 ]; then
            echo "### Inference completed successfully. ###"
            break
        else
            echo "### Inference failed with exit code $INFER_STATUS. Retrying... ###"
        fi
    done

    # Extract the output directory using the special delimiters
    OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o '### OUTPUT FILE:.* ###' | sed 's/### OUTPUT FILE: \(.*\) ###/\1/')
    echo "Got OUTPUT_FILE: $OUTPUT_FILE"

    while true; do
        echo "### Evaluating on $OUTPUT_FILE ... ###"
        OUTPUT_CONFIG_FILE="${OUTPUT_FILE%.jsonl}_config.json"
        export EVAL_SKIP_BUILD_ERRORS=true
        COMMAND="poetry run python ./evaluation/benchmarks/multi_swe_bench/scripts/eval/update_multi_swe_bench_config.py --input $OUTPUT_FILE --output $OUTPUT_CONFIG_FILE --dataset $EVAL_DATASET;
        poetry run python -m multi_swe_bench.harness.run_evaluation --config $OUTPUT_CONFIG_FILE
        "

        echo "Running command: $COMMAND"
        # Run the command
        eval $COMMAND
        EVAL_STATUS=$?
        if [ $EVAL_STATUS -eq 0 ]; then
            echo "### Evaluation completed successfully. ###"
            break
        else
            echo "### Evaluation failed with exit code $EVAL_STATUS. Retrying... ###"
        fi

        ./evaluation/utils/scripts/cleanup_remote_runtime.sh
    done

    # update the output with evaluation results
    echo "### Updating the output with evaluation results... ###"
    poetry run python evaluation/benchmarks/multi_swe_bench/scripts/eval/update_output_with_eval.py $OUTPUT_FILE

    echo "### Combining the final completions... ###"
    poetry run python evaluation/benchmarks/multi_swe_bench/scripts/eval/combine_final_completions.py $OUTPUT_FILE

    echo "### DONE for run $run_idx! ###"
    echo "You can find the final output at $(dirname $OUTPUT_FILE)/$FINAL_OUTPUT_FILE"
done
