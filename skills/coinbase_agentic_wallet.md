---
name: coinbase-agentic-wallet
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- x402
- coinbase wallet
- usdc
- paid api
- agentic wallet
- pay for api
---

You have access to the **Coinbase Agentic Wallet CLI** (`awal`) and can use it to autonomously discover and pay for HTTP APIs that implement the [x402 protocol](https://docs.cdp.coinbase.com/x402/welcome). Payments settle in USDC on Base, Polygon, or Solana via a Coinbase-managed embedded wallet -- no API keys, no manual onramp, no seed phrases.

## Quick reference

| Goal | Command |
|------|---------|
| Check status and auth | `npx awal status` |
| Sign in one time | `npx awal auth login <email>` then `npx awal auth verify <flowId> <otp>` |
| Get wallet address | `npx awal address` |
| Check USDC balance | `npx awal balance` (add `--chain solana` or `--chain polygon` for non-Base) |
| Search for paid APIs | `npx awal x402 bazaar search "<query>"` |
| Inspect payment terms | `npx awal x402 details <url>` |
| Call a paid API | `npx awal x402 pay <url>` |
| Send USDC | `npx awal send <amount> <recipient>` |
| Machine-readable output | Append `--json` |

<IMPORTANT>
- Always run `npx awal status` first to confirm auth before paid calls.
- Never put a private key in any `awal` command; the wallet is managed and only needs email/OTP sign-in.
- For autonomous workflows, prefer `--json` output and parse with `jq`.
- If a paid call fails with insufficient balance, surface a clear message to the user and ask before triggering an onramp.
</IMPORTANT>

## Example: discover and pay flow

```bash
# 1. Find candidate services
npx awal x402 bazaar search "weather forecast" --json | jq '.results[] | {url, price, description}'

# 2. Inspect one
npx awal x402 details https://api.example.com/forecast --json

# 3. Pay and call
npx awal x402 pay https://api.example.com/forecast --json
```

## When to use the MCP server instead

If you need a tool surface over MCP instead of shell commands, run the Coinbase
installer once:

```bash
npx @coinbase/payments-mcp install --client other
```

Then configure the generated bundle as an MCP server in `config.template.toml`
under `[mcp.stdio_servers]`:

```toml
[mcp.stdio_servers.coinbase_agentic_wallet]
command = "sh"
args = ["-c", "node \"$HOME/.payments-mcp/bundle.js\""]
```

## References

- Agentic Wallet welcome: https://docs.cdp.coinbase.com/agentic-wallet/welcome
- Agentic Wallet CLI quickstart: https://docs.cdp.coinbase.com/agentic-wallet/cli/quickstart
- x402 protocol: https://docs.cdp.coinbase.com/x402/welcome
