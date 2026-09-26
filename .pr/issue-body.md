### Problem or Motivation

Today it is hard to hand an external incident-analysis system (e.g. HolmesGPT or any RCA producer) into an OpenHands coding task. RCA tools produce structured findings — summary, evidence, suspected components/files, and a recommended action — but the only task input today is free-form text. Pasting raw JSON into the composer loses structure and gives the agent no instruction to verify the analysis against the repository before changing code.

### Desired Behavior

As a user (or an external system integrating with Agent Canvas), I can attach a structured RCA payload to a new conversation so that the coding agent receives the summary, evidence, suspected components/files, and recommended action as labeled context ahead of my task text, verifies it against the repository, and then acts on it.

Concretely:

- The home launcher offers an "Add RCA context" control that opens a paste-JSON modal, validates the payload (a non-empty `summary` is required; `evidence`, `suspected_components`, `suspected_files`, `recommended_action`, and `source` are optional; snake_case and camelCase keys are both accepted), and shows an attached chip that can be removed.
- Submitting with only RCA context and no typed text is allowed.
- The `/launch` deeplink accepts an optional base64-encoded `rca` param alongside the existing `plugins` and `message` params; plugins remain required, and an undecodable `rca` param shows the launch error.
- The RCA is composed into the first user message as a labeled "Root Cause Analysis Context" markdown block on both local and Cloud conversation creation, telling the agent to treat it as a hypothesis to verify against the repository before making changes.
- Ordinary tasks without RCA are unchanged.

### Acceptance Criteria

- [ ] A screenshot/video demonstrating the feature behavior is attached to the PR.
- [ ] Pasting a valid RCA JSON payload in the home import modal attaches it and the created conversation's first message contains the formatted context block ahead of the task text.
- [ ] Invalid JSON or a payload without a `summary` shows an inline error and does not attach.
- [ ] `/launch?plugins=<base64>&rca=<base64>` creates a conversation whose first message contains the RCA block; `?rca=` without `plugins` still errors.
- [ ] A normal task (no RCA) produces the same first message as before.
- [ ] Unit tests cover parsing/validation, message composition, deeplink decoding, and the home UI flow.

### Alternatives Considered

- Free-form paste of the RCA JSON as task text: works but loses field structure, is error-prone, and does not instruct the agent to verify the analysis before editing.
- A new Agent Server field for RCA: unnecessary — composing the block into the existing initial message keeps the Canvas integration generic and requires no backend contract change or `minimumAgentServer` bump.

### Additional Context

The payload schema is intentionally tool-agnostic (`summary`, `evidence`, `suspected_components`, `suspected_files`, `recommended_action`, `source`) so HolmesGPT or any other external investigation tool can emit it. A `buildRcaLaunchPath` helper produces ready-to-use deeplink URLs.
