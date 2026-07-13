# Secret-aware settings LLM smoke evidence

Generated on 2026-07-13 UTC for PR #15110 / issue #15116.

This package closes the previously incomplete LLM smoke item without external
provider access. The harness stands up a local authenticated OpenAI-compatible
HTTP endpoint, saves an org LLM profile through the enterprise application API,
restarts the app, reloads the profile through `SaasSettingsStore` and
`resolve_profile_llm`, then invokes the recovered profile through the SDK/LiteLLM
completion path.

## Files

- `run_llm_profile_smoke.py`: reproducible harness.
- `main-llm-smoke.json`: sanitized result for current main
  `3949e1cc17d9443f1f4ef7d34d428baf065cd919`.
- `pr-llm-smoke.json`: sanitized result for merged PR product head
  `a441ca5a486aa1b3a2be6273ba752d776ea56d16`.

## Flow

1. Reset a dedicated PostgreSQL database and run enterprise migrations.
2. Start the enterprise app against PostgreSQL and Redis.
3. Save profile `local-openai-compatible-smoke` through
   `POST /api/organizations/{org_id}/profiles/{profile_name}` with model
   `openai/gpt-4o-mini`, a local `/v1` base URL, and a generated sentinel API key.
4. Verify the app API returns a masked secret before and after restart.
5. Inspect the database with sanitized checks only.
6. Reload settings through the real service/model path after restart.
7. Invoke the reloaded profile through the SDK/LiteLLM completion path.

## Results

| Checkout | Result | Key evidence |
| --- | --- | --- |
| Current main `3949e1cc17d9443f1f4ef7d34d428baf065cd919` | PASS for this narrower LLM smoke | Missing-auth and wrong-key requests returned 401. Post-restart LiteLLM request used auth fingerprint `c56a1c9c692c03bd`, matching the stored sentinel fingerprint. Completion returned `local-smoke-completion-ok`. API responses were masked and raw secret checks were false. Storage remained opaque/non-JSON, so this run is not evidence that main has the PR's field-level leaf storage behavior. |
| Merged PR product head `a441ca5a486aa1b3a2be6273ba752d776ea56d16` | PASS | Missing-auth and wrong-key requests returned 401. Post-restart LiteLLM request used auth fingerprint `03d6f8f025c17cbf`, matching the stored sentinel fingerprint. Completion returned `local-smoke-completion-ok`. API responses were masked, raw secret checks were false, and `llm_profiles` was parseable JSON with an encrypted leaf marker plus visible non-secret profile markers. |

Current main passed this local LLM post-restart invocation path, so this package
does not claim a red baseline for that narrower item. The previously captured
red baseline remains the nested MCP/profile leaf-storage preservation failure on
current main.

## Sanitization

The harness writes only SHA-256 prefixes and booleans for generated keys. It
asserts that the raw sentinel API key, wrong API key, app API key, and JWT secret
are absent from each JSON artifact before saving it. Server logs were not
included in this package.
