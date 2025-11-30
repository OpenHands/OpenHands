#!/usr/bin/env bash
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
MAX_ITER=$5
NUM_WORKERS=$6
DATASET=$7
SPLIT=$8
N_RUNS=$9
MODE=${10}


if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  echo "MAX_ITER not specified, use default 100"
  MAX_ITER=100
fi

if [ -z "$RUN_WITH_BROWSING" ]; then
  echo "RUN_WITH_BROWSING not specified, use default false"
  RUN_WITH_BROWSING=false
fi


if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default princeton-nlp/SWE-bench_Lite"
  DATASET="swefficiency/swefficiency"
fi

if [ -z "$SPLIT" ]; then
  echo "SPLIT not specified, use default test"
  SPLIT="test"
fi

if [ -z "$MODE" ]; then
  MODE="swe"
  echo "MODE not specified, use default $MODE"
fi

if [ -n "$EVAL_CONDENSER" ]; then
  echo "Using Condenser Config: $EVAL_CONDENSER"
else
  echo "No Condenser Config provided via EVAL_CONDENSER, use default (NoOpCondenser)."
fi

export RUN_WITH_BROWSING=$RUN_WITH_BROWSING
echo "RUN_WITH_BROWSING: $RUN_WITH_BROWSING"

get_openhands_version

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "MODE: $MODE"
echo "EVAL_CONDENSER: $EVAL_CONDENSER"

# Default to NOT use Hint
if [ -z "$USE_HINT_TEXT" ]; then
  export USE_HINT_TEXT=false
fi
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION"
# if not using Hint, add -no-hint to the eval note
if [ "$USE_HINT_TEXT" = false ]; then
  EVAL_NOTE="$EVAL_NOTE-no-hint"
fi

if [ "$RUN_WITH_BROWSING" = true ]; then
  EVAL_NOTE="$EVAL_NOTE-with-browsing"
fi

if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi
# if mode != swe, add mode to the eval note
if [ "$MODE" != "swe" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${MODE}"
fi
# Add condenser config to eval note if provided
if [ -n "$EVAL_CONDENSER" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"
fi

# export RUNTIME="remote"
# export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
export NO_CHANGE_TIMEOUT_SECONDS=900 # 15 minutes

function run_eval() {
  local eval_note="${1}"
  COMMAND="poetry run python evaluation/benchmarks/swefficiency/run_infer.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $eval_note \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE"

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  # Run the command
  eval $COMMAND
}

unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, use default $N_RUNS"
fi

# Skip runs if the run number is in the SKIP_RUNS list
# read from env variable SKIP_RUNS as a comma separated list of run numbers
SKIP_RUNS=(${SKIP_RUNS//,/ })
for i in $(seq 1 $N_RUNS); do
  if [[ " ${SKIP_RUNS[@]} " =~ " $i " ]]; then
    echo "Skipping run $i"
    continue
  fi
  current_eval_note="$EVAL_NOTE-run_$i"
  echo "EVAL_NOTE: $current_eval_note"
  run_eval $current_eval_note
done

checkout_original_branch
