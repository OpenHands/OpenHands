# Reproduction evidence for #16893

`before.png` and `after.png` are the same vitest run on either side of the
fix, rendered from the captured output rather than screen-grabbed.

`before.png` restores the two hooks from `upstream/main` and runs the new
test: it fails with `expected 2 to be 4`, meaning the query key carried two
segments (`["skills", null]`) where four are needed once the active backend's
id and orgId are appended. `after.png` restores this branch's version and the
same test passes.

Reproduce with:

```
git checkout upstream/main -- src/hooks/query/use-skills.ts src/hooks/query/use-conversation-skills.ts
npx vitest run src/hooks/query/use-skills.test.tsx   # fails
git checkout HEAD -- src/hooks/query/use-skills.ts src/hooks/query/use-conversation-skills.ts
npx vitest run src/hooks/query/use-skills.test.tsx   # passes
```
