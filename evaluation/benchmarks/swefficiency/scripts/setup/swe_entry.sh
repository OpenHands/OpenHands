#!/usr/bin/env bash

set -e

# assert user name is `root`
if [ "$USER" != "root" ]; then
    echo "Error: This script is intended to be run by the 'root' user only." >&2
    exit 1
fi

source ~/.bashrc

SWEUTIL_DIR=/swe_util

# Create logs directory
LOG_DIR=/openhands/logs
mkdir -p $LOG_DIR && chmod 777 $LOG_DIR

# FIXME: Cannot read SWE_INSTANCE_ID from the environment variable
# SWE_INSTANCE_ID=django__django-11099
if [ -z "$SWE_INSTANCE_ID" ]; then
    echo "Error: SWE_INSTANCE_ID is not set." >&2
    exit 1
fi

# Read the swe-bench-test-lite.json file and extract the required item based on instance_id
item=$(jq --arg INSTANCE_ID "$SWE_INSTANCE_ID" '.[] | select(.instance_id == $INSTANCE_ID)' $SWEUTIL_DIR/eval_data/instances/swe-bench-test-lite.json)

if [[ -z "$item" ]]; then
  echo "No item found for the provided instance ID."
  exit 1
fi

CONDA_ENV_NAME=$(echo "$item" | jq -r '.repo + "__" + .version | gsub("/"; "__")')

echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"

SWE_TASK_DIR=/openhands/swe_tasks
mkdir -p $SWE_TASK_DIR
# Dump test_patch to /workspace/test.patch
echo "$item" | jq -r '.test_patch' > $SWE_TASK_DIR/test.patch
# Dump patch to /workspace/gold.patch
echo "$item" | jq -r '.patch' > $SWE_TASK_DIR/gold.patch
# Dump the item to /workspace/instance.json except for the "test_patch" and "patch" fields
echo "$item" | jq 'del(.test_patch, .patch)' > $SWE_TASK_DIR/instance.json

# Clear the workspace
rm -rf /workspace/*
# Copy repo to workspace
if [ -d /workspace/$CONDA_ENV_NAME ]; then
    rm -rf /workspace/$CONDA_ENV_NAME
fi
cp -r $SWEUTIL_DIR/eval_data/testbeds/$CONDA_ENV_NAME /workspace

# Reset swe-bench testbed and install the repo
. $SWEUTIL_DIR/miniforge3/etc/profile.d/conda.sh
conda config --set changeps1 False
conda config --append channels conda-forge
conda activate swe-bench-eval

mkdir -p $SWE_TASK_DIR/reset_testbed_temp
mkdir -p $SWE_TASK_DIR/reset_testbed_log_dir
SWE_BENCH_DIR=/swe_util/OH-SWE-bench
output=$(
    export PYTHONPATH=$SWE_BENCH_DIR && \
    cd $SWE_BENCH_DIR && \
    python swebench/harness/reset_swe_env.py \
    --swe_bench_tasks $SWEUTIL_DIR/eval_data/instances/swe-bench-test.json \
    --temp_dir $SWE_TASK_DIR/reset_testbed_temp \
    --testbed /workspace \
    --conda_path $SWEUTIL_DIR/miniforge3 \
    --instance_id $SWE_INSTANCE_ID \
    --log_dir $SWE_TASK_DIR/reset_testbed_log_dir \
    --timeout 900 \
    --verbose
)

REPO_PATH=$(echo "$output" | awk -F': ' '/repo_path:/ {print $2}')
TEST_CMD=$(echo "$output" | awk -F': ' '/test_cmd:/ {print $2}')
echo "Repo Path: $REPO_PATH"
echo "Test Command: $TEST_CMD"

echo "export SWE_BENCH_DIR=\"$SWE_BENCH_DIR\"" >> ~/.bashrc
echo "export REPO_PATH=\"$REPO_PATH\"" >> ~/.bashrc
echo "export TEST_CMD=\"$TEST_CMD\"" >> ~/.bashrc

if [[ "$REPO_PATH" == "None" ]]; then
    echo "Error: Failed to retrieve repository path. Tests may not have passed or output was not as expected." >&2
    exit 1
fi

# Activate instance-specific environment
. $SWEUTIL_DIR/miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

set +e
