# Severity Levels

Defines how the Reviewer Agent classifies issues found during code review.

## Level Definitions

| Level | Label | Meaning | Required Action |
|-------|-------|---------|----------------|
| CRITICAL | 🔴 | Security vulnerability, data loss risk, or hardcoded secret | Must fix before merge |
| HIGH | 🟡 | Functional bug or significant quality/performance issue | Should fix before merge |
| MEDIUM | 🔵 | Maintainability concern, code smell, or minor issue | Consider fixing |
| LOW | ⚪ | Style suggestion or optional improvement | Optional |

## Classification Rules

### Critical

Any of the following is automatically CRITICAL:

- Hardcoded API keys, passwords, tokens, or connection strings in source code
- SQL injection: string concatenation in SQL/NoSQL queries with unsanitized input
- Command injection: unsanitized input in `os.system()`, `subprocess`, or shell calls
- Stored/Reflected XSS: unsanitized user input rendered as HTML
- Path traversal: unsanitized file path from user input
- Authentication/authorization bypass
- Hardcoded cryptographic keys or IVs

### High

The following are typically HIGH:

- Missing error handling that could lead to data loss or crash
- N+1 query pattern in database access
- Missing pagination on unbounded queries
- Function exceeding 100 lines
- File exceeding 1000 lines
- PR template fields missing (Why, Summary, How to Test, Type)
- Direct mutation of function parameters
- Frontend component importing API client directly (bypassing TanStack Query)

### Medium

The following are typically MEDIUM:

- Function between 50-100 lines (could be smaller)
- Nesting depth of 4-5 levels
- Missing type hints on public functions (Python)
- Unused imports or variables
- Non-descriptive variable names
- Missing inline comments for non-obvious logic
- Lockfile not updated after dependency change

### Low

The following are typically LOW:

- Minor style deviations from project conventions
- Trailing whitespace or formatting issues
- Missing Chinese/English spacing in comments
- Suggestive improvements without concrete impact
- Documentation typos or formatting

## Verdict Determination

| Condition | Verdict |
|-----------|---------|
| Any CRITICAL issue | **Request Changes** |
| Any HIGH issue | **Request Changes** |
| Only MEDIUM/LOW issues | **Approve** (with suggestions) |
| No issues found | **Approve** |
