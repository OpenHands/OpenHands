# Where this workflow comes from

This is an independently written adaptation of ideas from poteto's MIT-licensed
[pstack](https://github.com/cursor/plugins/tree/main/pstack) and the
[cursor-team-kit](https://github.com/cursor/plugins/tree/main/cursor-team-kit)
control skills:

- [create-verification-skill](https://github.com/cursor/plugins/blob/main/pstack/skills/create-verification-skill/SKILL.md):
  interview the repo, then generate launch / doctor / drive / evidence / cleanup
  and a feature map, and prove one recipe end to end before handing it over.
- [The feature-map example](https://github.com/cursor/plugins/tree/main/pstack/skills/create-verification-skill/references/feature-map-example):
  one file per feature with four sections, stable IDs, and every step written
  against a `control-<app>` CLI so it is a literal, rerunnable command.
- [maintain-verification-skill](https://github.com/cursor/plugins/blob/main/pstack/skills/maintain-verification-skill/SKILL.md):
  index hygiene, a parallel source wave, one serial live pass, and triage into
  map drift, harness gaps and product bugs.
- [Build the Lever](https://github.com/cursor/plugins/blob/main/pstack/skills/principle-build-the-lever/SKILL.md)
  and [control-ui](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/control-ui/SKILL.md):
  build the tool that drives and proves the work; one structural action per
  step, fresh state before each assertion, stable handles over coordinates.
- [cli-for-agents](https://github.com/cursor/plugins/blob/main/cli-for-agent/skills/cli-for-agents/SKILL.md):
  non-interactive flags, layered `--help` with examples, actionable errors,
  idempotent commands, machine-readable output.

OpenHands-specific decisions:

- `control-openhands` wraps the repository's own production launcher
  (`bin/agent-canvas.mjs`) instead of a new stack, and isolates each run with a
  private `HOME`, state directory, session key and port block.
- A long-lived browser daemon keeps one page between commands and records page
  errors, console errors and failed requests continuously, so short CLI calls do
  not lose the transitions between them.
- LLM profiles are configured from an environment key (`llm preset deepseek`),
  validated with a one-token completion first, so a CI agent can run
  model-backed recipes without typing secrets into argv or the UI log.
- Weekly maintenance adds what OpenHands needs beyond pstack: a frozen
  BASE/TARGET, a PR-intent ledger, and intent kept separate from runtime results,
  with multi-repository attribution (Canvas, Agent Server/SDK, automation).
- pstack's guide asks for the maintenance pass "at least once a day" but
  prescribes no tiers, time budget, sharding or map-to-test linkage.
  [daily.md](daily.md) adds those for a map this size: static checks first
  (`map check`, `map coverage`, `map testids`), the changed families from
  `map affected` (the `Source:` lines as a change-to-feature index, in the
  spirit of `tests/e2e/mock-llm/test-mapping.json`), a smoke row per family,
  a weekly rotation, and `evidence report --baseline` as the day's verdict.
  The Playwright suites under `tests/e2e/` are not referenced from the map
  (they run differently and prove a commit, not this run); the behaviors they
  assert are mapped as sub-features with their own live recipes instead.
- Not imported: Cursor plugin configuration, model routing,
  `disable-model-invocation`, autonomous shipping or merge authority, and the
  zero-pixel-difference rule (deliberate weekly product changes are expected).

These are contributor workflows for this repository under `.agents/skills`
([AgentSkills format](https://docs.openhands.dev/sdk/guides/skill)), not public
catalog skills from `OpenHands/extensions`. They complement the public
`qa-changes` skill (a verdict on one PR) and [the release testing
matrix](../../../../docs/TESTING_MATRIX.md) (installer × OS × agent), and do not
replace either. The first version of this map was seeded from the September 2026
UI audit in OpenHands/OpenHands#17435 and OpenHands/OpenHands#17569.
