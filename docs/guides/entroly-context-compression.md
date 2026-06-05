# Entroly Integration: Context Compression for OpenHands Agents

Reduce LLM API costs by 70–95% when running OpenHands agents on large codebases.

## Problem

OpenHands agents send full repository context to LLM providers on every request. For large repos (500+ files), this means 100K+ tokens per request — most of which is irrelevant to the current task.

## Solution

[Entroly](https://github.com/juyterman1000/entroly) is a local context compression proxy that sits between OpenHands and the LLM provider:

```
OpenHands Agent  ──►  Entroly (local)  ──►  LLM Provider
                      │
                      ├─ Rank files by query relevance
                      ├─ Select optimal subset (knapsack)
                      ├─ Compress noisy context (reversible)
                      ├─ Align cache prefix (provider discounts)
                      └─ Verify reply (WITNESS guard)
```

## Setup

### 1. Install Entroly

```bash
pip install entroly
```

### 2. Start the proxy

```bash
entroly proxy --port 9377
```

### 3. Configure OpenHands to use the proxy

Set the LLM base URL in your OpenHands configuration:

```toml
[llm]
base_url = "http://localhost:9377/v1"
```

Or via environment variable:

```bash
export LLM_BASE_URL="http://localhost:9377/v1"
```

All LLM requests from OpenHands agents will now be automatically compressed.

### 4. Verify

```bash
entroly verify-claims
```

No API key required. Writes `.entroly_verification.json`.

## How It Works

- **BM25 + entropy + dependency graph** ranks every file by relevance
- **Knapsack optimization** packs the most valuable files under a token budget
- **CCR handles** ensure exact recovery of compressed content
- **Cache alignment** stabilizes prefixes for provider discounts (Anthropic 90%, OpenAI 50%)
- **WITNESS** checks answers against supplied evidence ($0, ~3ms)

## Results

| Metric | Result |
|---|---|
| Token reduction (large repos) | 70–95% |
| Accuracy (NeedleInAHaystack) | 100% retained |
| Hallucination detection (HaluEval-QA) | 0.844 AUROC |
| WITNESS latency | ~3ms, $0 |

## Links

- **GitHub**: [github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly)
- **License**: Apache-2.0
- Local-first, no outbound analytics by default
