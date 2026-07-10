# External Supervisor Pattern

## What is the external supervisor pattern?

The **external supervisor pattern** is an architectural approach for governing
what an autonomous coding agent is permitted to do, by placing policy
enforcement in a process that the agent **cannot influence**.

In this pattern, a dedicated *supervisor* sits between an agent caller (for
example, an automation runner, a CLI, or an MCP host) and the agent runtime.
Every action the agent requests is mediated by the supervisor, which decides
whether the action is allowed before it reaches the runtime or the host
filesystem. The supervisor is "external" because it runs outside the agent's
own execution context — the agent has no ability to disable, rewrite, or bypass
it, even if the agent is otherwise capable of arbitrary code execution.

This is distinct from, and complementary to, the sandbox-based isolation that
OpenHands already provides:

- OpenHands runs agents inside a **sandbox** (such as a Docker container) that
  isolates the execution environment and limits filesystem access to mounted
  directories (for example, the `PROJECTS_PATH` bind-mount described in the
  project `README.md`). The sandbox answers the question *"where does the agent
  run?"*
- An external supervisor answers a different question: *"which specific actions
  is the agent allowed to take, and are they within policy?"* It enforces
  fine-grained, auditable rules over commands and file changes regardless of
  where the runtime executes.

The two layers compose: the sandbox contains the blast radius of a misbehaving
agent, while the supervisor constrains and records what the agent is permitted
to attempt in the first place.

## Motivation

Autonomous agents that can run shell commands and edit files carry real risk.
The OpenHands `README.md` explicitly warns that running the agent server
without a sandbox means *"the agent will have full access to your filesystem."*
Even inside a sandbox, an agent may be instructed — or may decide on its own —
to run destructive commands, read sensitive files, or push changes outside the
intended scope of a task.

A supervisor addresses these concerns by making the policy explicit and
independent of the agent's own judgment:

- **Tamper resistance.** Because the supervisor is not part of the agent's
  process, an agent that gains the ability to edit its own configuration cannot
  weaken the rules that govern it.
- **Auditable by default.** Every approved or denied action can be recorded as
  structured evidence, producing a trail that is useful for review, compliance,
  and post-incident analysis.
- **Defense in depth.** Policy enforcement does not rely on a single boundary.
  If a sandbox escape or a runtime misconfiguration occurs, the supervisor
  still gates the actions that reach the host.

## Pattern architecture

```
+-------------------+        +---------------------+        +-----------------+
|  Agent caller     |        |  External supervisor|        |  Agent runtime  |
|  (CLI / MCP host /| -----> |  - allowlist check  | -----> |  (sandbox /     |
|   automation)     | action |  - path confinement  | action |   local / remote)|
+-------------------+        |  - sensitive block   |        +-----------------+
                             |  - audit record      |
                             +---------------------+
```

The caller submits an action (typically a shell command or a file mutation) to
the supervisor. The supervisor evaluates the action against its policy and
either forwards the approved action to the runtime or rejects it. In both
cases the decision is logged. The runtime never receives an action the
supervisor has not cleared.

A supervisor is typically exposed over a narrow, well-defined interface. Two
common shapes are:

- A **local bridge** (for example, an MCP server) that the agent's host calls
  instead of calling the runtime directly. The bridge is the only path to the
  runtime, so all actions are mediated.
- A **sidecar** that wraps the runtime's command-execution entry point,
  intercepting commands before they are dispatched.

In both shapes the invariant is the same: there is no unmediated path from the
caller to the runtime.

## Security boundaries

A supervisor enforces policy through several concrete, composable boundaries.
The boundaries below are the ones that most directly reduce risk for coding
agents; a given implementation may implement a subset.

### Workspace confinement

All file access is restricted to a configured workspace root. The supervisor
rejects any path that resolves outside this root, including paths that reach
outside through traversal (`..`), symbolic links, or absolute paths. This
ensures an agent operating on a repository cannot read or modify files the
operator did not intend to expose — for example, dotfiles, SSH keys, or
neighboring projects.

Confinement is enforced by resolving the target path to its canonical form and
verifying it remains within the workspace root, rather than by trusting the
path string the agent supplied.

### Command allowlists

The supervisor permits only commands that match an explicit allowlist.
Matching is exact and structural, not a substring or pattern match, so that an
allowlisted entry for `git status` does not also permit `git status; rm -rf /`.
Arguments are matched as well as the command name, so the same binary can be
allowed with one set of arguments and denied with another.

An allowlist stands in contrast to a denylist: a denylist must anticipate
every dangerous command and will inevitably miss some, whereas an allowlist
fails closed — anything not explicitly permitted is rejected. This is the
safer default for an autonomous agent that may receive untrusted instructions.

### Sensitive-file blocking

Even within the workspace root, certain filenames are refused outright because
they routinely hold secrets or state that an agent should not touch. This
includes files such as `.env`, credential files, SSH keys, cookies, and
browser-state stores. The block applies to both reads and writes, so an agent
cannot exfiltrate secrets by reading them or corrupt state by overwriting them.

### Auditable evidence

Each decision — approved or denied — is recorded as a structured artifact that
includes the action, the decision, the matched rule, and a timestamp. This
evidence is written by the supervisor, not by the agent, so it cannot be
altered by the agent after the fact. The record is useful for review, for
correlating agent behavior with task outcomes, and for tightening policy over
time.

## Relationship to the OpenHands sandbox

OpenHands' sandbox model (described in `openhands/app_server/sandbox/README.md`)
manages the *lifecycle and isolation* of the environment an agent runs in:
creating, starting, stopping, and destroying containers, with multiple backend
support (Docker, Remote, Local) and user-scoped access control. That model
controls where execution happens and how the environment is provisioned.

An external supervisor operates at a different layer. It does not provision or
isolate environments; it governs the actions permitted within or in front of
them. The two are designed to compose:

- The **sandbox** limits the blast radius if an agent misbehaves — a
  compromised container cannot reach the host directly.
- The **supervisor** limits what the agent is permitted to attempt, and keeps
  an independent record, so that even an agent operating inside a correctly
  configured sandbox is constrained to a declared policy.

Operators who already run OpenHands with a Docker sandbox can add a supervisor
in front of the runtime without changing the sandbox configuration. Operators
who run without a sandbox (the "Without a Sandbox" and "From Source" options in
the `README.md`, which grant the agent full filesystem access) gain the most
from a supervisor, because it provides a policy boundary that the absent
container does not.

## Example implementation: PatchWarden

[PatchWarden](https://github.com/jiezeng2004-design/PatchWarden) is one
implementation of this pattern. It is a local MCP bridge that mediates agent
commands against an allowlist, confines paths to a configured workspace root,
blocks sensitive filenames, and writes structured task evidence including
before/after Git snapshots.

PatchWarden is referenced here only as a concrete instance of the pattern. The
pattern itself is implementation-agnostic: any component that satisfies the
invariants above — unmediated access is impossible, policy is enforced outside
the agent, decisions are auditable — implements the external supervisor
pattern. The boundary definitions (workspace confinement, command allowlists,
sensitive-file blocking, auditable evidence) are the load-bearing parts; the
specific tool, language, or transport is not.

## When to use this pattern

Consider an external supervisor when any of the following apply:

- The agent runs **without a sandbox**, or with a sandbox the operator does not
  fully trust, and a policy boundary is needed beyond what the runtime enforces.
- The agent operates on **shared or sensitive hosts** where uncommanded file
  changes or destructive commands would cause harm beyond the task.
- **Auditable evidence** of what the agent attempted — and what was denied — is
  required for review, compliance, or incident response.
- The operator wants a **fail-closed** default (anything not explicitly
  permitted is rejected) rather than relying on the agent's own judgment or a
  denylist that must anticipate every dangerous action.

The pattern adds a hop between the caller and the runtime, so it is not free.
For trusted, fully-sandboxed, ephemeral environments where the operator accepts
the agent's judgment, the added boundary may be unnecessary. The trade-off is
between the latency and operational cost of the supervisor and the reduced risk
of an unmediated, un-audited agent action.

## References

- OpenHands sandbox model: [`openhands/app_server/sandbox/README.md`](../openhands/app_server/sandbox/README.md)
- OpenHands runtime and sandbox warnings: [`README.md`](../README.md) (sections "Without a Sandbox" and "From Source")
- External integration event flow: [`enterprise/doc/architecture/external-integrations.md`](../enterprise/doc/architecture/external-integrations.md)
- PatchWarden (example implementation): https://github.com/jiezeng2004-design/PatchWarden
