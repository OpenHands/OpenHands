# OpenHands Fork Operator Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, secret-safe operator CLI and CI gate that prepares, validates, and starts this OpenHands fork without changing upstream application behavior.

**Architecture:** Implement a Python standard-library CLI under `scripts/` with pure readiness/provider helpers and thin system probes. Validate it through a dependency-free `unittest` suite and a path-filtered GitHub Actions workflow. Document the supported local, Codespaces, WSL, and VPS workflows.

**Tech Stack:** Python 3.12 standard library, `unittest`, GitHub Actions, existing OpenHands `make` targets.

## Global Constraints

- Do not modify upstream backend or frontend behavior.
- Do not add dependencies or regenerate lockfiles.
- Do not print, persist, or commit provider secrets.
- Docker is the default runtime; local runtime must be selected explicitly.
- ChatGPT browser authentication is not an API credential source.
- All system checks must return actionable `pass`, `warning`, or `error` results.

---

### Task 1: Add failing operator contract tests

**Files:**
- Create: `tests/unit/test_openhands_operator.py`

**Interfaces:**
- Consumes: future module loaded from `scripts/openhands_operator.py`.
- Produces: executable behavior contracts for `parse_version`, `validate_provider`, `build_child_environment`, `bootstrap_workspace`, `render_report`, and `main`.

- [ ] **Step 1: Write tests for version parsing and provider validation**
- [ ] **Step 2: Write tests proving secret values never appear in human or JSON output**
- [ ] **Step 3: Write tests for OpenCode Go to `LLM_*` child-environment mapping**
- [ ] **Step 4: Write tests for workspace/config bootstrap and sanitized dry-run output**
- [ ] **Step 5: Add a temporary CI workflow and verify the tests fail because the implementation file is absent**
- [ ] **Step 6: Commit the red test state**

Run:

```bash
python -m unittest tests.unit.test_openhands_operator -v
```

Expected before implementation: failure while importing `scripts/openhands_operator.py`.

### Task 2: Implement the operator CLI

**Files:**
- Create: `scripts/openhands_operator.py`

**Interfaces:**
- Produces:
  - `CheckResult(name: str, status: str, message: str)`
  - `ReadinessReport(results: list[CheckResult])`
  - `parse_version(value: str) -> tuple[int, int, int] | None`
  - `validate_provider(env: Mapping[str, str], mode: str, require_provider: bool) -> list[CheckResult]`
  - `build_child_environment(env: Mapping[str, str], provider_mode: str) -> dict[str, str]`
  - `bootstrap_workspace(repo_root: Path, workspace: Path, create_config: bool) -> list[str]`
  - `render_report(report: ReadinessReport, as_json: bool) -> str`
  - `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Implement dataclasses and deterministic report rendering**
- [ ] **Step 2: Implement semantic version parsing and command probes**
- [ ] **Step 3: Implement runtime, workspace, port, and provider readiness checks**
- [ ] **Step 4: Implement secret-safe OpenCode Go child-environment mapping**
- [ ] **Step 5: Implement `doctor`, `bootstrap`, and `start` subcommands**
- [ ] **Step 6: Run the unit tests and fix only implementation defects**
- [ ] **Step 7: Run syntax compilation and CLI help checks**
- [ ] **Step 8: Commit the green implementation**

Run:

```bash
python -m py_compile scripts/openhands_operator.py
python -m unittest tests.unit.test_openhands_operator -v
python scripts/openhands_operator.py --help
```

Expected: all commands exit `0`.

### Task 3: Add the permanent readiness CI gate

**Files:**
- Create: `.github/workflows/fork-operator-readiness.yml`

**Interfaces:**
- Consumes: operator script and unit tests.
- Produces: a required-quality signal for pushes and pull requests touching the operator layer.

- [ ] **Step 1: Trigger only on operator script, tests, workflow, and fork operations documentation**
- [ ] **Step 2: Use GitHub-authored checkout and Python setup actions**
- [ ] **Step 3: Run `py_compile`, unit tests, and CLI help**
- [ ] **Step 4: Commit the workflow**
- [ ] **Step 5: Inspect the pull-request workflow run and logs**

### Task 4: Document operation and security boundaries

**Files:**
- Create: `docs/fork/OPERATIONS.md`

**Interfaces:**
- Consumes: CLI behavior from Task 2.
- Produces: copy-paste startup instructions for WSL/Codespaces/VPS operators.

- [ ] **Step 1: Document `doctor`, `bootstrap`, and one-command `start --bootstrap --build`**
- [ ] **Step 2: Document generic and OpenCode Go provider variables without real values**
- [ ] **Step 3: Document JSON/strict modes for CI and troubleshooting**
- [ ] **Step 4: Document remote-access, Docker, filesystem, and credential risks**
- [ ] **Step 5: Explicitly separate ChatGPT browser login from API authentication**
- [ ] **Step 6: Commit documentation**

### Task 5: Final review and pull request

**Files:**
- Review all files from Tasks 1-4.

- [ ] **Step 1: Compare the branch against `main` and confirm no upstream core files changed**
- [ ] **Step 2: Re-run the complete operator verification commands**
- [ ] **Step 3: Check the design success criteria one by one**
- [ ] **Step 4: Open a draft pull request with exact validation evidence and remaining manual runtime steps**
- [ ] **Step 5: Review CI results; fix failures and re-run until green or report the precise blocker**
