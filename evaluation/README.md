# Evaluation

This folder contains code and resources to run experiments and evaluations.

## For Benchmark Users

### Setup

Before starting evaluation, follow the instructions [here](https://github.com/OpenHands/OpenHands/blob/main/Development.md) to setup your local development environment and LLM.

Once you are done with setup, you can follow the benchmark-specific instructions in each subdirectory of the [evaluation directory](#supported-benchmarks).
Generally these will involve running `run_infer.py` to perform inference with the agents.

### Implementing and Evaluating an Agent

To add an agent to OpenHands, you will need to implement it in the [agenthub directory](https://github.com/OpenHands/OpenHands/tree/main/openhands/agenthub). There is a README there with more information.

To evaluate an agent, you can provide the agent's name to the `run_infer.py` program.

### Evaluating Different LLMs

OpenHands in development mode uses `config.toml` to keep track of most configuration.
**IMPORTANT: For evaluation, only the LLM section in `config.toml` will be used. Other configurations, such as `save_trajectory_path`, are not applied during evaluation.**

Here's an example configuration file you can use to define and use multiple LLMs:

```toml
[llm]
# IMPORTANT: add your API key here, and set the model to the one you want to evaluate
model = "gpt-4o-2024-05-13"
api_key = "sk-XXX"

[llm.eval_gpt4_1106_preview_llm]
model = "gpt-4-1106-preview"
api_key = "XXX"
temperature = 0.0

[llm.eval_some_openai_compatible_model_llm]
model = "openai/MODEL_NAME"
base_url = "https://OPENAI_COMPATIBLE_URL/v1"
api_key = "XXX"
temperature = 0.0
```

### Configuring Condensers for Evaluation

For benchmarks that support condenser configuration (like SWE-Bench), you can define multiple condenser configurations in your `config.toml` file. A condenser is responsible for managing conversation history to maintain context while staying within token limits - you can learn more about how it works [here](https://www.all-hands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents):

```toml
# LLM-based summarizing condenser for evaluation
[condenser.summarizer_for_eval]
type = "llm"
llm_config = "haiku"  # Reference to an LLM config to use for summarization
keep_first = 2        # Number of initial events to always keep
max_size = 100        # Maximum size of history before triggering summarization

# Recent events condenser for evaluation
[condenser.recent_for_eval]
type = "recent"
keep_first = 2        # Number of initial events to always keep
max_events = 50       # Maximum number of events to keep in history
```

You can then specify which condenser configuration to use when running evaluation scripts, for example:

```bash
EVAL_CONDENSER=summarizer_for_eval \
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 500 100 1 princeton-nlp/SWE-bench_Verified test
```

The name is up to you, but should match a name defined in your `config.toml` file. The last argument in the command specifies the condenser configuration to use. In this case, `summarizer_for_eval` is used, which refers to the LLM-based summarizing condenser as defined above.

If no condenser configuration is specified, the 'noop' condenser will be used by default, which keeps the full conversation history.

For other configurations specific to evaluation, such as `save_trajectory_path`, these are typically set in the `get_config` function of the respective `run_infer.py` file for each benchmark.

### Enabling LLM-Based Editor Tools

The LLM-Based Editor tool (currently supported only for SWE-Bench) can be enabled by setting:
```bash
export ENABLE_LLM_EDITOR=true
```

You can set the config for the Editor LLM as:
```toml
[llm.draft_editor]
base_url = "http://localhost:9002/v1"
model = "hosted_vllm/lite_coder_qwen_editor_3B"
api_key = ""
temperature = 0.7
max_input_tokens = 10500
max_output_tokens = 10500
```

## Supported Benchmarks

The OpenHands evaluation harness supports a wide variety of benchmarks across [software engineering](#software-engineering), [web browsing](#web-browsing), [miscellaneous assistance](#misc-assistance), and [real-world](#real-world) tasks.

### Software Engineering

- SWE-Bench: [`evaluation/benchmarks/swe_bench`](./benchmarks/swe_bench)
- HumanEvalFix: [`evaluation/benchmarks/humanevalfix`](./benchmarks/humanevalfix)
- BIRD: [`evaluation/benchmarks/bird`](./benchmarks/bird)
- BioCoder: [`evaluation/benchmarks/biocoder`](./benchmarks/biocoder)
- ML-Bench: [`evaluation/benchmarks/ml_bench`](./benchmarks/ml_bench)
- APIBench: [`evaluation/benchmarks/gorilla`](./benchmarks/gorilla/)
- ToolQA: [`evaluation/benchmarks/toolqa`](./benchmarks/toolqa/)
- AiderBench: [`evaluation/benchmarks/aider_bench`](./benchmarks/aider_bench/)
- Commit0: [`evaluation/benchmarks/commit0_bench`](./benchmarks/commit0_bench/)
- DiscoveryBench: [`evaluation/benchmarks/discoverybench`](./benchmarks/discoverybench/)
- TerminalBench: [`evaluation/benchmarks/terminal_bench`](./benchmarks/terminal_bench)

### Web Browsing

- WebArena: [`evaluation/benchmarks/webarena`](./benchmarks/webarena/)
- MiniWob++: [`evaluation/benchmarks/miniwob`](./benchmarks/miniwob/)
- Browsing Delegation: [`evaluation/benchmarks/browsing_delegation`](./benchmarks/browsing_delegation/)

### Misc. Assistance

- GAIA: [`evaluation/benchmarks/gaia`](./benchmarks/gaia)
- GPQA: [`evaluation/benchmarks/gpqa`](./benchmarks/gpqa)
- AgentBench: [`evaluation/benchmarks/agent_bench`](./benchmarks/agent_bench)
- MINT: [`evaluation/benchmarks/mint`](./benchmarks/mint)
- Entity deduction Arena (EDA): [`evaluation/benchmarks/EDA`](./benchmarks/EDA)
- ProofWriter: [`evaluation/benchmarks/logic_reasoning`](./benchmarks/logic_reasoning)
- ScienceAgentBench: [`evaluation/benchmarks/scienceagentbench`](./benchmarks/scienceagentbench)

### Real World

- TheAgentCompany: [`evaluation/benchmarks/the_agent_company`](./benchmarks/the_agent_company)

## Result Visualization

Check [this huggingface space](https://huggingface.co/spaces/OpenHands/evaluation) for visualization of existing experimental results.

You can start your own fork of [our huggingface evaluation outputs](https://huggingface.co/spaces/OpenHands/evaluation) and submit a PR of your evaluation results to our hosted huggingface repo via PR following the guide [here](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions#pull-requests-and-discussions).

## For Benchmark Developers

To learn more about how to integrate your benchmark into OpenHands, check out [tutorial here](https://docs.all-hands.dev/usage/how-to/evaluation-harness). Briefly,

- Each subfolder contains a specific benchmark or experiment. For example, [`evaluation/benchmarks/swe_bench`](./benchmarks/swe_bench) should contain
all the preprocessing/evaluation/analysis scripts.
- Raw data and experimental records should not be stored within this repo.
- For model outputs, they should be stored at [this huggingface space](https://huggingface.co/spaces/OpenHands/evaluation) for visualization.
- Important data files of manageable size and analysis scripts (e.g., jupyter notebooks) can be directly uploaded to this repo.
