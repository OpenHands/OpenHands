<h1 align="center"> Training Software Engineering Agents and Verifiers with SWE-Gym </h1>

A Multi-SWE-bench implementation of SWE-Gym.

<p align="center">
  <a href="https://www.jiayipan.com/" style="text-decoration: none;">Jiayi Pan<sup>*,1</sup></a>,
  <a href="https://xwang.dev/" style="text-decoration: none;">Xingyao Wang<sup>*,2</sup></a>,
  <a href="https://www.phontron.com/" style="text-decoration: none;">Graham Neubig<sup>3</sup></a>,
  <a href="https://www.cs.toronto.edu/~ndjaitly/" style="text-decoration: none;">Navdeep Jaitly<sup>4</sup></a>,
  <a href="https://blender.cs.illinois.edu/hengji.html" style="text-decoration: none;">Heng Ji<sup>2</sup></a>,
  <a href="https://www.alanesuhr.com/" style="text-decoration: none;">Alane Suhr<sup>^,1</sup></a>,
  <a href="https://dreasysnail.github.io/" style="text-decoration: none;">Yizhe Zhang<sup>^,4</sup></a>
</p>

<p align="center">
  <sup>1</sup>UC Berkeley, <sup>2</sup>UIUC, <sup>3</sup>CMU, <sup>4</sup>Apple </br>
  <sub><sup>*</sup>Equal contribution, <sup>^</sup>Equal supervision</sub>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2412.21139">📃 Paper</a>
•
<a href="https://huggingface.co/SWE-Gym" >🤗 Data & Models</a>
</p>

We present **SWE-Gym**, the first environment for training real-world software engineering agents.
We use it to train strong LM agents that achieve state-of-the-art open results on SWE-Bench, with early, promising scaling characteristics as we increase training and inference-time compute.

<p align="center">
  <img src="https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/teaser.jpg?raw=true" width="100%" alt="teaser">
</p>

---
# Run SWE-Gym with OpenHands

The process of running SWE-Gym is very similar to how you'd run SWE-Bench evaluation.


1. First, clone OpenHands repo `git clone https://github.com/OpenHands/OpenHands.git`
2. Then setup the repo following [Development.md](https://github.com/OpenHands/OpenHands/blob/main/Development.md)
3. Then you can simply serve your own model as an OpenAI compatible endpoint, put those info in config.toml. You can do this by following instruction [here](../../README.md#setup).
4. And then simply do the following to sample for 16x parallelism:

```bash
export ALLHANDS_API_KEY=ah-yourkey  # You don't need to set this when running these in local docker container
./evaluation/benchmarks/multi_swe_bench/scripts/rollout_swegym.sh llm.mymodel-temp05 'train-t05' 16
```

NOTE: SWE-Gym sampling with parallelism is currently only tested with AllHands RemoteRuntime (limited beta). Fill [this form](https://docs.google.com/forms/d/e/1FAIpQLSckVz_JFwg2_mOxNZjCtr7aoBFI2Mwdan3f75J_TrdMS1JV2g/viewform) to apply for access.


5. When `rollout_swegym.sh` finishes, you will get a file called `output.with_completions.jsonl.gz`. Then you can use [`./scripts/swegym/convert_data.ipynb`](./scripts/swegym/convert_data.ipynb) to convert them into SFT data format.

## Running the Jupyter Notebook

To run the data conversion notebook, follow these steps:

1. Navigate to the OpenHands repository root:
```bash
cd openhands_repo
```

2. Set the PYTHONPATH and start Jupyter notebook:
```bash
PYTHONPATH=$(pwd) jupyter notebook
```

3. In the Jupyter interface, navigate to `evaluation/benchmarks/swe_bench/scripts/swegym/convert_data.ipynb`

4. Update the file paths in the notebook:
   - Set `FILE_PATHS` to point to your `output.with_completions.jsonl.gz` files
   - Set `YOUR_OUTPUT_FOLDER` to your desired output directory

5. Run the notebook cells sequentially to process your data and generate the SFT training format.

---
# More info about SWE-Gym

Progress in agents for software engineering has been limited by the lack of training environments that both include rigorous verification for reinforcement learning and cover the expansive tasks encountered in real-world repository-level engineering.

We introduce SWE-Gym: An Open Environment for Training Software Engineering Agents & Verifiers.
Our baselines achieve new open SOTA - 32%/26% on SWE-Bench Verified/Lite, with promising scaling trends.

![SWE-Gym Scaling](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/scaling.jpg?raw=true)
*SWE-Gym enables scalable improvements for software engineering agents at both training and inference time. Our current results is primarily bottlenecked by training and inference compute, rather than the size of our environment.*

## SWE-Gym Environment

We create SWE-Gym, the first environment for training SWE agents, with **2.4K real tasks from 11 Python repos** & a Lite split of 234 instances. SWE-Gym combines real-world Python tasks, repository context, executable environments, and test verification to train agents for solving software engineering problems.

![SWE-Gym Repo Distribution](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/swe-gym.jpg?raw=true)


## SWE-Gym trains LMs as agents

When fine-tuned on less than 500 agent-environment interaction trajectories sampled from it from GPT-4o and Claude 3.5 Sonnet, we achieve **+14%** absolute gains on SWE-Bench Verified with an 32B LM-powered OpenHands agent.

![OpenHands Performance diff before and after training](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/oh-agent.jpg?raw=true)


## SWE-Gym enables self-improvement

SWE-Gym is also effective across agent scaffolds. With rejection sampling fine-tuning and MoatlessTools scaffold, our 32B and 7B models achieve 20% and 10% respectively on SWE-Bench Lite through self-improvement.

<p align="center">
  <img src="https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/ml-agent.jpg?raw=true" width="80%" alt="Moatless self-improvement">
</p>



## SWE-Gym enables inference-time scaling

SWE-Gym enables inference-time scaling through verifiers trained on agent trajectories.
These verifiers identify most promising solutions via best-of-n selection, together with our learned agents, they achieve 32%/26% on SWE-Bench Verified/Lite, a new open SoTA.


![Inference Time Scaling for Moatless Agent](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/inference-ml.jpg?raw=true)
*Inference Time Scaling for Moatless Agent*

![Inference Time Scaling for OpenHands Agent](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/inference-oh.jpg?raw=true)
*Inference Time Scaling for OpenHands Agent*


## Our baselines on SWE-Gym shows strong scaling trends

Lastly, our ablations reveal strong scaling trends - performance is now bottlenecked by train and inference compute, rather than the size of our dataset. Pushing and improving these scaling trends further is an exciting direction for future work.

![](https://github.com/SWE-Gym/SWE-Gym/blob/main/assets/images/scaling.jpg?raw=true)

## Reproducing Results
**The Dataset**

To access SWE-Gym dataset, checkout our huggingface hub page [SWE-Gym](https://huggingface.co/SWE-Gym)

The environment constants are currently saved at [SWE-Bench-Fork](https://github.com/SWE-Gym/SWE-Bench-Fork)

We also have pre-built docker images for each instance under [xingyaoww/sweb.eval.x86_64](https://hub.docker.com/search?q=xingyaoww%2Fsweb.eval.x86_64.) prefix at docker hub.


## 📚 Citation

```bibtex
@misc{pan2024trainingsoftwareengineeringagents,
      title={Training Software Engineering Agents and Verifiers with SWE-Gym},
      author={Jiayi Pan and Xingyao Wang and Graham Neubig and Navdeep Jaitly and Heng Ji and Alane Suhr and Yizhe Zhang},
      year={2024},
      eprint={2412.21139},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2412.21139},
}
```
