For inference,
```
poetry run python evaluation/benchmarks/nl2repo/run_infer.py   --agent-cls CodeActAgent   --llm-config llm.your-llm-config   --max-iterations 500   --eval-num-workers 10  --dataset /workspace/dataset/nl2repo.jsonl   --split train
```

For evaluation,
```
poetry run python evaluation/benchmarks/nl2repo/eval_infer.py --dataset /workspace/dataset/nl2repo/nl2repo.jsonl --eval-output-dir <path-to-your-result-dir>
```