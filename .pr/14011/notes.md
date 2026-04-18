# PR 14011 QA evidence

## Summary
- Rebased the branch onto current `main` and resolved the new settings-query-key conflicts.
- While validating the ESLint rule, I found one additional raw settings query key in `frontend/src/hooks/mutation/use-delete-git-providers.ts` and migrated it to `SETTINGS_QUERY_KEYS.personal(...)`.

## Local validation
- `cd frontend && npm run lint`
- `cd frontend && npm run build`
- `cd frontend && printf 'const query = { queryKey: ["settings", "personal", null] };\nexport default query;\n' | npx eslint --stdin --stdin-filename src/routes/user-settings.tsx`

## Notes
- The final ESLint smoke test is expected to fail with `Use SETTINGS_QUERY_KEYS helpers instead of raw settings query key arrays.`; that failure confirms the guardrail works.
- Logs are attached in `.pr/14011/logs/`.
