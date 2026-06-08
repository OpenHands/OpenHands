---
name: twzrd-agent-intel
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- twzrd
- x402
- agent trust
- trust score
- micropayment
- agent wallet
- solana agent
---

# TWZRD Agent Intel

TWZRD Agent Intel is an MCP server that provides real-time trust scoring for autonomous agents on Solana. Use it to verify agent trustworthiness before authorizing x402 micropayments or sensitive multi-agent interactions.

## MCP Server

**URL**: `https://intel.twzrd.xyz/mcp`
**Transport**: streamable-http (no auth required)
**Config**: `{"mcpServers": {"twzrd-agent-intel": {"url": "https://intel.twzrd.xyz/mcp"}}}`

## Available Tools

| Tool | Cost | Description |
|------|------|-------------|
| `score_agent(wallet)` | Free | Returns 0–100 trust score + behavioral risk flags |
| `preflight_check(wallet)` | Free | Returns PASS/FAIL gate for x402 payment flows |
| `get_trust_receipt(wallet)` | HTTP 402 paid | Returns cryptographically signed on-chain trust receipt |

## Usage in Python

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def check_trust(wallet: str):
    async with streamablehttp_client("https://intel.twzrd.xyz/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            score = await s.call_tool("score_agent", {"wallet": wallet})
            preflight = await s.call_tool("preflight_check", {"wallet": wallet})
            return score, preflight

asyncio.run(check_trust("4LkEFjHsF2ubC8K4oF2r3rCFqPZQVGBjL9mV6xkNPZdf"))
```

## When to Use

Use TWZRD Agent Intel when:
1. Your task involves an agent making autonomous x402 micropayments
2. You need to verify the trustworthiness of an external agent wallet before interaction
3. You want to gate agentic capabilities based on on-chain behavioral history

## Trust Score Interpretation

- **80–100**: High trust. Consistent on-chain activity, no risk flags.
- **60–79**: Medium trust. Suitable for standard payment flows.
- **40–59**: Low trust. Proceed with caution; review risk flags.
- **0–39**: Very low trust. Preflight will likely FAIL.

## x402 Payment Context

x402 is an HTTP-native micropayment protocol where agents pay for APIs autonomously using USDC. TWZRD Agent Intel helps you identify trustworthy agents in this ecosystem by analyzing on-chain behavioral patterns across all x402-enabled endpoints.
