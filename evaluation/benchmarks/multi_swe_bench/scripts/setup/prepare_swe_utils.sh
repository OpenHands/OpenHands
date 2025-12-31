#!/bin/bash

set -e
EVAL_WORKSPACE="evaluation/benchmarks/swe_bench/eval_workspace"
mkdir -p $EVAL_WORKSPACE

# 1. Prepare REPO
echo "==== Prepare SWE-bench repo ===="
OH_SWE_BENCH_REPO_PATH="https://github.com/OpenHands/SWE-bench.git"
OH_SWE_BENCH_REPO_BRANCH="eval"
git clone -b $OH_SWE_BENCH_REPO_BRANCH $OH_SWE_BENCH_REPO_PATH $EVAL_WORKSPACE/OH-SWE-bench

# 2. Prepare DATA
echo "==== Prepare SWE-bench data ===="
EVAL_IMAGE=ghcr.io/openhands/eval-swe-bench:builder_with_conda
EVAL_WORKSPACE=$(realpath $EVAL_WORKSPACE)
chmod +x $EVAL_WORKSPACE/OH-SWE-bench/swebench/harness/prepare_data.sh
if [ -d $EVAL_WORKSPACE/eval_data ]; then
    rm -r $EVAL_WORKSPACE/eval_data
fi
docker run \
    -v $EVAL_WORKSPACE:/workspace \
    -w /workspace \
    -u $(id -u):$(id -g) \
    -e HF_DATASETS_CACHE="/tmp" \
    --rm -it $EVAL_IMAGE \
    bash -c "cd OH-SWE-bench/swebench/harness && /swe_util/miniforge3/bin/conda run -n swe-bench-eval ./prepare_data.sh && mv eval_data /workspace/"
