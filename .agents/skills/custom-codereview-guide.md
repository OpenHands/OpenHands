---
name: custom-codereview-guide
description: Repo-specific code review guidelines for OpenHands/agent-canvas. Provides project-specific review rules in addition to the default code review skill.
triggers:
- /codereview
---

# OpenHands/agent-canvas Code Review Guidelines

You are an expert code reviewer for the **OpenHands/agent-canvas** repository. This skill provides repo-specific review guidelines. Be direct but constructive.

## Review Decisions

You have permission to **APPROVE** or **COMMENT** on PRs. Do not use REQUEST_CHANGES.

**Mandatory:** Always submit exactly one PR review object before finishing. If you found no actionable issues, post a short **APPROVE** review rather than ending silently without posting a review. If you found actionable issues or concerns, post a **COMMENT** review.

### Review decision policy (eval / benchmark risk)

Do **NOT** submit an **APPROVE** review when the PR changes agent behavior or anything
that could plausibly affect benchmark/evaluation performance.

Examples include: prompt templates, tool calling/execution, planning/loop logic,
memory/condenser behavior, terminal/stdin/stdout handling, or evaluation harness code.

If a PR is in this category (or you are uncertain), leave a **COMMENT** review and
explicitly flag it for a human maintainer to decide after running lightweight evals.

### Default approval policy

**Default to APPROVE**: If your review finds no issues at "important" level or higher,
approve the PR. Minor suggestions or nitpicks alone are not sufficient reason to
withhold approval.

**IMPORTANT:** If you determine a PR is worth merging **and it is not in the eval-risk
category above**, you should approve it. Don’t just say a PR is "worth merging" or
"ready to merge" without actually submitting an approval. Your words and actions should
be consistent.

### When to APPROVE

Examples of straightforward and low-risk PRs you should approve (non-exhaustive):

- **Configuration changes**: Adding models to config files, updating CI/workflow settings
- **CI/Infrastructure changes**: Changing runner types, fixing workflow paths, updating job configurations
- **Cosmetic changes**: Typo fixes, formatting, comment improvements, README updates
- **Documentation-only changes**: Docstring updates, clarifying notes, API documentation improvements
- **Simple additions**: Adding entries to lists/dictionaries following existing patterns
- **Test-only changes**: Adding or updating tests without changing production code
- **Dependency updates**: Version bumps with passing CI, unless the updated package is newer than the repo's 7-day freshness guardrail described in the Security section below

### When NOT to APPROVE - Blocking Issues

**DO NOT APPROVE** PRs that have any of the following issues:

- **Package version bumps in non-release PRs**: If any `pyproject.toml` file has changes to the `version` field (e.g., `version = "1.12.0"` → `version = "1.13.0"`), and the PR is NOT explicitly a release PR (title/description doesn't indicate it's a release), **DO NOT APPROVE**. Version numbers should only be changed in dedicated release PRs managed by maintainers.
  - Check: Look for changes to `version = "..."` in any `*/pyproject.toml` files
  - Exception: PRs with titles like "release: v1.x.x" or "chore: bump version to 1.x.x" from maintainers
- **Too-new dependency uploads**: If a dependency bump pulls in a package uploaded within the repo's 7-day freshness window, **DO NOT APPROVE**. See the Security section below for the exact review instructions and the Dependabot / `tool.uv.exclude-newer` caveat.

Examples:
- A PR adding a new model to `resolve_model_config.py` or `verified_models.py` with corresponding test updates
- A PR adding documentation notes to docstrings clarifying method behavior (e.g., security considerations, bypass behaviors)
- A PR changing CI runners or fixing workflow infrastructure issues (e.g., standardizing runner types to fix path inconsistencies)

### When to COMMENT

Use COMMENT when you have feedback or concerns:

- Issues that need attention (bugs, security concerns, missing tests)
- Suggestions for improvement
- Questions about design decisions
- Minor style preferences

If there are significant issues, leave detailed comments explaining the concerns—but let a human maintainer decide whether to block the PR.

## Security

### Dependency freshness / supply-chain guardrail

This repository intentionally uses a workspace-wide `uv` resolver guardrail:

- Root `pyproject.toml`: `[tool.uv] exclude-newer = "7 days"`

**Important:** Dependabot does **not** currently honor that `uv` guardrail when it opens `uv.lock` update PRs for this repo's workspace setup. A Dependabot PR can therefore bump to a version that was uploaded **less than 7 days ago**, even though a local `uv lock` would normally exclude it.

When reviewing dependency update PRs (`uv.lock`, `pyproject.toml`, `requirements*.txt`, etc.), explicitly check for **too-new package uploads**:

1. Check the package upload timestamp on the package index.
2. For `uv.lock`, use the per-file `upload-time` metadata in the changed package entry.
3. Treat `upload-time` as the upload time of that specific distribution file to the package index (for example, the wheel uploaded to PyPI) — not the Git tag time or GitHub release time.
4. Compare that timestamp against the current date and the repo's 7-day freshness window.

If the updated package was uploaded **within the last 7 days**, treat it as a real security / supply-chain concern:

- Do **NOT** approve the PR.
- Leave a **COMMENT** review that clearly calls out the package name, version, upload time, and that it is newer than the repo's 7-day guardrail.
- Explain that this can happen because Dependabot currently ignores `tool.uv.exclude-newer` for this repo's workspace updates.
- Ask a human maintainer to decide whether to wait until the package ages past the guardrail or to merge intentionally despite the freshness risk.

## Core Principles

1. **Simplicity First**: Question complexity. If something feels overcomplicated, ask "what's the use case?" and seek simpler alternatives. Features should solve real problems, not imaginary ones.

2. **Pragmatic Testing**: Test what matters. Avoid duplicate test coverage. Don't test library features (e.g., `BaseModel.model_dump()`). Focus on the specific logic implemented in this codebase.

3. **Type Safety**: Avoid `# type: ignore` - treat it as a last resort. Fix types properly with assertions, proper annotations, or code adjustments. Prefer explicit type checking over `getattr`/`hasattr` guards.

4. **Backward Compatibility**: Evaluate breaking change impact carefully. Consider API changes that affect existing users, removal of public fields/methods, and changes to default behavior.

## What to Check

- **Complexity**: Over-engineered solutions, unnecessary abstractions, complex logic that could be refactored
- **Testing**: Duplicate test coverage, tests for library features, missing edge case coverage. For code that writes to disk, verify that tests cover the **persistence round-trip** (write → close → reopen → verify), not just in-memory state
- **Type Safety**: `# type: ignore` usage, missing type annotations, `getattr`/`hasattr` guards, mocking non-existent arguments
- **Breaking Changes**: API changes affecting users, removed public fields/methods, changed defaults
- **Code Quality**: Code duplication, missing comments for non-obvious decisions, inline imports (unless necessary for circular deps)
- **Repository Conventions**: Use `pyright` not `mypy`, put fixtures in `conftest.py`, avoid `sys.path.insert` hacks
- **Event Type Deprecation**: Changes to event types (Pydantic models used in serialization) must handle deprecated fields properly
- **Thread Safety**: New methods in `LocalConversation` that read or write `self._state` must use `with self._state:` — see the [Concurrency](#concurrency---localconversation-state-lock) section below
- **Persistence Paths**: Code that computes persistence directories must not double-append the conversation hex — see the [Persistence Paths](#persistence-path-construction) section below
- **Server-Side Cleanup**: Endpoints that create persistent state (directories, files) must have rollback logic for partial failures — see the [Server Error Handling](#server-side-error-handling) section below
- **Cross-File Data Flow**: When new code calls existing APIs (constructors, factory methods), trace 1–2 levels into those APIs to verify the caller uses them correctly. Bugs often hide at layer boundaries where the caller's assumptions don't match the callee's behavior

## Event Type Deprecation - Critical Review Checkpoint

When reviewing PRs that modify event types (e.g., `TextContent`, `Message`, `Event`, or any Pydantic model used in event serialization), **DO NOT APPROVE** until the following are verified:

### Required for Removing/Deprecating Fields

1. **Model validator present**: If a field is being removed from an event type with `extra="forbid"`, there MUST be a `@model_validator(mode="before")` that uses `handle_deprecated_model_fields()` to remove the deprecated field before validation. Otherwise, old events will fail to load.

2. **Tests for backward compatibility**: The PR MUST include tests that:
   - Load an old event format (with the deprecated field) successfully
   - Load a new event format (without the deprecated field) successfully
   - Verify both can be loaded in sequence (simulating mixed conversations)

3. **Test naming convention**: The version in the test name should be the **LAST version** where a particular event structure exists. For example, if `enable_truncation` was removed in v1.11.1, the test should be named `test_v1_10_0_...` (the last version with that field), not `test_v1_8_0_...` (when it was introduced). This avoids duplicate tests and clearly documents when a field was last present.

**Important**: Deprecated field handlers are **permanent** and should never be removed. They ensure old conversations can always be loaded.

### Example Pattern (Required)

```python
from openhands.sdk.utils.deprecation import handle_deprecated_model_fields

class MyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Deprecated fields that are silently removed for backward compatibility
    # when loading old events. These are kept permanently.
    _DEPRECATED_FIELDS: ClassVar[tuple[str, ...]] = ("old_field_name",)

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_fields(cls, data: Any) -> Any:
        """Remove deprecated fields for backward compatibility with old events."""
        return handle_deprecated_model_fields(data, cls._DEPRECATED_FIELDS)
```

### Why This Matters

Production systems resume conversations that may contain events serialized with older SDK versions. If the SDK can't load old events, users will see errors like:

```
pydantic_core.ValidationError: Extra inputs are not permitted
```

**This is a production-breaking change.** Do not approve PRs that modify event types without proper backward compatibility handling and tests.

## Frontend API Access Conventions

These two rules are enforced by the CI test `src/api/no-direct-agent-server-calls.test.ts`.
**Flag any PR that introduces a violation** -- these are correctness bugs, not style nits.

### Rule 1 -- All agent-server calls must use `@openhands/typescript-client`

**DO NOT APPROVE** a PR that introduces raw `axios`, `fetch`, or the shared `openHands`
axios instance to call an agent-server endpoint (`/api/*`, `/server_info`). All such
calls must go through typed client classes from `@openhands/typescript-client`,
instantiated with options from `getAgentServerClientOptions()` or
`getAgentServerHttpClientOptions()` in `src/api/agent-server-client-options.ts`.

Forbidden patterns (caught by the CI guard):
- `openHands.<method>(...)` -- shared axios instance
- `createHttpClient(...)` -- creates a raw HTTP client
- `axios(...)` / `axios.get/post/etc.(...)` (except in the two allowed files)
- `fetch('/api/...')` or `fetch(\`${host}/api/...\`)`

Correct pattern:
```ts
new ConversationClient(getAgentServerClientOptions()).getConversation(id)
new FileClient(getAgentServerClientOptions()).downloadTextFile(path)
new ServerClient(getAgentServerHttpClientOptions()).getServerInfo()
new RemoteWorkspace(getAgentServerClientOptions()).gitChanges({ ref: "HEAD" })
```

Allowed exceptions (explicitly listed in `ALLOWED_AD_HOC_HTTP_FILES`):
- `src/api/automation-service/automation-service.api.ts`
- `src/api/cloud/proxy.ts`

If a PR adds a new file to `ALLOWED_AD_HOC_HTTP_FILES` without a strong reason,
flag it -- the allowlist should not grow casually.

### Rule 2 -- All cloud backend calls must go through `callCloudProxy`

**DO NOT APPROVE** a PR that issues a direct browser `fetch` or `axios` call to the
cloud backend (`app.all-hands.dev`) or a cloud runtime sandbox
(`*.prod-runtime.all-hands.dev`). Both origins block CORS from `localhost`. Cloud calls
must go through `callCloudProxy()` in `src/api/cloud/proxy.ts`, which routes them
server-side through `/api/cloud-proxy` on the local agent-server.

Correct pattern -- cloud:
```ts
callCloudProxy({ backend, method: "GET", path: "/api/v1/app-conversations/search?..." })
```

Correct pattern -- cloud runtime sandbox (use `hostOverride` + `authMode: "session-api-key"`):
```ts
callCloudProxy({
  backend,
  method: "GET",
  hostOverride: buildHttpBaseUrl(conversationUrl),
  path: `/api/conversations/${id}`,
  authMode: "session-api-key",
  sessionApiKey,
})
```

Standard branch structure every cloud-aware service method should follow:
```ts
if (getActiveBackend().backend.kind === "cloud") {
  return callCloudProxy({ backend: active, ... });
}
// local path: typed typescript-client
return new ConversationClient(getAgentServerClientOptions()).someMethod(...);
```

Missing the `hostOverride` on a runtime-sandbox call is a silent bug: the proxy
will target `backend.host` (the cloud API) instead of the actual runtime URL.
Flag any `callCloudProxy` call that targets a runtime URL without `hostOverride`.

## SDK Architecture Conventions

These conventions codify patterns that are easy to violate when adding new features. Each was learned from a real bug.

### Concurrency - LocalConversation State Lock

`LocalConversation` protects mutable state with a FIFOLock accessed via `with self._state:`. **Every** method that reads or writes `self._state.events`, `self._state.stats`, `self._state.agent_state`, `self._state.activated_knowledge_skills`, or any other mutable field on `ConversationState` must hold this lock. There are currently ~13 call sites using this pattern.

When reviewing a PR that adds a new method to `LocalConversation`:
1. Check whether it accesses any `self._state.*` field.
2. If yes, verify the access is inside a `with self._state:` block.
3. If not, flag it — the method is unsafe for concurrent use with `run()`.

### Persistence Path Construction

`BaseConversation.get_persistence_dir(base, conversation_id)` returns `str(Path(base) / conversation_id.hex)`. The `LocalConversation.__init__` constructor calls this automatically when `persistence_dir` is provided.

**Rule:** Callers that pass `persistence_dir` to `LocalConversation()` must pass only the **base directory** (e.g., `/data/conversations/`). The constructor appends the conversation hex. Passing a pre-constructed full path (e.g., `/data/conversations/abc123`) causes double-appending: `/data/conversations/abc123/abc123`.

When reviewing code that creates a new `LocalConversation` (fork, resume, migration):
1. Check what value is passed as `persistence_dir`.
2. Verify it does **not** already include the conversation ID hex.

### Server-Side Error Handling

Server endpoints in `conversation_service.py` that create persistent state (writing directories, files, or calling `fork()` which writes to disk) and then perform follow-up operations (like `_start_event_service`) must handle partial failure.

**Pattern:** If the follow-up operation fails, clean up the already-written persistent state so it doesn't become an orphaned directory that confuses future startups.

```python
# Good: rollback on failure
fork_dir = self.conversations_dir / fork_conv_id.hex
try:
    fork_event_service = await self._start_event_service(fork_stored)
except Exception:
    safe_rmtree(fork_dir)
    raise
```

When reviewing server endpoints that create conversations or persistent artifacts:
1. Identify the "point of no return" where state is written to disk.
2. Check that subsequent operations are wrapped in try/except with cleanup.
3. For client-supplied IDs, verify there's a duplicate check before creating state (return 409 Conflict if taken).

## E2E Test Label Triage

The `e2e-tests` label triggers the mock-LLM E2E and Docker E2E test suites on a
PR. When reviewing, use your judgement to decide whether the changes could
benefit from full end-to-end testing. If the PR doesn't already have the label
and you think it should, add it:

```bash
gh pr edit <PR_NUMBER> --add-label "e2e-tests" --repo OpenHands/agent-canvas
```

Mention in your review body that you added the label (one sentence is enough).
When in doubt, add it — running the tests is cheap, missing a regression is not.
Skip it for obviously safe changes like docs-only, pure styling, or CI config
tweaks.

If the PR touches an area that lacks mock-LLM E2E coverage and would benefit
from it, suggest adding a test in `tests/e2e/mock-llm/` as part of the PR or a
follow-up.

## What NOT to Comment On

Do not leave comments for:

- **Nitpicks**: Minor style preferences, optional improvements, or "nice-to-haves" that don't affect correctness or maintainability
- **Good behavior observed**: Don't comment just to praise code that follows best practices - this adds noise. Simply approve if the code is good.
- **Suggestions for additional tests on simple changes**: For straightforward PRs (config changes, model additions, etc.), don't suggest adding test coverage unless tests are clearly missing for new logic
- **Obvious or self-explanatory code**: Don't ask for comments on code that is already clear
- **`.pr/` directory artifacts**: Files in the `.pr/` directory are temporary PR-specific documents (design notes, analysis, scripts) that are automatically cleaned up when the PR is approved. Do not comment on their presence or suggest removing them.

If a PR is approvable, just approve it. Don't add "one small suggestion" or "consider doing X" comments that delay merging without adding real value.

## Communication Style

- Be direct and concise - don't over-explain
- Use casual, friendly tone ("lgtm", "WDYT?", emojis are fine 👀)
- Ask questions to understand use cases before suggesting changes
- Suggest alternatives, not mandates
- Approve quickly when code is good ("LGTM!")
- Use GitHub suggestion syntax for code fixes
