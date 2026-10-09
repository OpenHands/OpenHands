# `llm preset deepseek` repoints the `default` agent profile

Both screenshots come from one run of `main` @ `ff1a1d6`, taken on
2026-10-09. Onboarding had just pinned the `default` agent profile to
`deepseek-chat`.

- [before-main-preset-agents.png](before-main-preset-agents.png): after
  `main`'s `llm preset deepseek`. `default` still names `deepseek-chat`, and
  `DELETE /api/profiles/deepseek-chat` answers `409`.
- [after-branch-preset-agents.png](after-branch-preset-agents.png): after this
  branch's `llm preset deepseek`, which reports `repointed`. `default` names
  `deepseek-flash`, and the delete answers `200`.
