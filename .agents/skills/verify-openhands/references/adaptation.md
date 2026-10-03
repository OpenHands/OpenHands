# Why this is an OpenHands workflow

This is an independently written adaptation of ideas from poteto's MIT-licensed
[pstack](https://github.com/cursor/plugins/tree/6ed0f7a9504f577d7529064103cecce9be7dfc5e/pstack),
read at that revision:

- [Create verification skill](https://github.com/cursor/plugins/blob/6ed0f7a9504f577d7529064103cecce9be7dfc5e/pstack/skills/create-verification-skill/SKILL.md): interview the repo; launch/doctor/drive/evidence/cleanup; prove one real recipe.
- [Maintain verification skill](https://github.com/cursor/plugins/blob/6ed0f7a9504f577d7529064103cecce9be7dfc5e/pstack/skills/maintain-verification-skill/SKILL.md): source coverage then a serial live pass; separate map drift, harness gaps, and product bugs.
- [Feature-map example](https://github.com/cursor/plugins/tree/6ed0f7a9504f577d7529064103cecce9be7dfc5e/pstack/skills/create-verification-skill/references/feature-map-example): stable feature IDs, user entry points, action/result recipes, and gotchas.
- [Prove it works](https://github.com/cursor/plugins/blob/6ed0f7a9504f577d7529064103cecce9be7dfc5e/pstack/skills/principle-prove-it-works/SKILL.md): inspect real artifacts, not self-reports or compilation alone.

Keep progressive disclosure, evidence that survives cleanup, and single ownership
of a live UI. Do not import Cursor's plugin configuration, model routing,
`disable-model-invocation`, control-ui tooling, autonomous shipping authority, or
its strict zero-pixel-difference rule for deliberate weekly product changes.

OpenHands uses [.agents/skills and AgentSkills frontmatter](https://docs.openhands.dev/sdk/guides/skill).
These are contributor workflows for this repository, not additions to the public
skill catalog owned by `OpenHands/extensions`. Distinct names avoid overriding the
public `qa-changes` skill. Use it for a specific PR's stated goal; use this map to
navigate the app and the weekly skill to reconcile a whole main-branch interval.

The additional OpenHands requirements are: backend/capability-aware coverage,
real SDK/automation paths, safe fixture and credential handling, reload/lifecycle
checks, multi-repository attribution, and explicit PR-intent evidence. Missing
credentials or inaccessible Cloud features remain gaps, never synthetic passes.
The PR #17435 audit seeded the known-issue recipes; it is historical evidence,
not a standing certification of current main.
