# Checks at implementation commit 794ec7c6898f28b6f358df93140cc829d6f2fa42

Environment: macOS, Node 24.22.1, committed package-lock installed with npm ci.

| Command | Result |
|---|---|
| `npm run lint` | Pass; type checking and formatting pass; 272 existing warnings, zero errors |
| `npm run build` | Pass |
| `npm run build:lib` | Pass |
| `npx vitest run __tests__/components/conversation-events/chat/group-events.test.ts __tests__/components/conversation-events/chat/messages-stable-event-props.test.tsx __tests__/hooks/chat/use-chat-input-events.test.tsx --maxWorkers=2` | 55 tests / 3 files pass |
| `npm test -- --maxWorkers=4` | 873 files pass, 2 fail; 9,968 tests pass, 15 fail, 1 skipped, 7 todo; 574.48 seconds |

All full-suite failures are in unchanged Docker shell tests. Separate runs against clean main `dce9d017abe9ed3d362bc77e0cd3fff2e4931f22` reproduced all 15 failures:

- `__tests__/scripts/docker-automation-kv-secret.test.ts`: 13 fail, 17 pass. The failing assertions expect empty stderr; the local Bash startup emits `.bashrc: line 7: /.local/bin/env: No such file or directory`.
- `__tests__/scripts/docker-session-key-policy.test.ts`: both tests fail with shell exit 127 in the main control as well as this branch.

No shell configuration or Docker source was changed to make these unrelated tests pass. CI on the PR supplies the final remote-platform result.

Regression-first checks failed on main and pass after the fix: the real Messages/group/summary path read action IDs 41,408 times for one group and 33,728 for 16 groups with 128 retained actions; both now stay below 1,280. Three ordinary composer input/resize calls used to broadcast three store updates with unchanged suggestion visibility; they now broadcast none. The same test also confirms actual visibility transitions still notify.

The [real-backend before/after check](verification.md) covers typing, focus, incoming updates and history retention on desktop and phone. The [current-head virtualization review](virtualization-review/README.md) is a separate paired check of PR #17820. Neither substitutes for live LLM, Cloud or ACP session coverage, and no bounded-memory claim is made.
