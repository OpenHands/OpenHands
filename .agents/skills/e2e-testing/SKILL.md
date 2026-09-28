---
name: e2e-testing
description: This skill should be used when the user asks to "add an E2E test", "run live E2E", "run mock-LLM tests", "debug Playwright CI", "test the Docker image", or changes tests/e2e, Playwright configs, E2E workflows, artifacts, or test reporting.
---

# End-to-End Testing

Use the correct test layer and preserve the security and production-fidelity boundaries between ordinary browser tests, mock-LLM suites, Docker image tests, and live LLM-backed QA.

## Workflow

1. Read `references/guide.md` before changing any E2E framework file or debugging an E2E failure.
2. Select the cheapest suite that exercises the real changed boundary.
3. Keep live credentials isolated to trusted execution steps and prevent analytics or sensitive artifacts.
4. Diagnose CI failures from workflow summaries and Playwright artifacts before modifying locators or timeouts.
5. Follow the repository testing rules: cover behavior through real code paths, minimize cases, and mock underlying services rather than hooks.

## Reference

- **`references/guide.md`** — Live and mock-LLM architecture, Docker mode, local commands, CI/reporting security, artifact debugging, common failure patterns, and general testing rules.
