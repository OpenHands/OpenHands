# Issue #16927 — Real launcher lifecycle evidence (PR #17438)

Reviewer request: demonstrate the actual launcher path on the current head —
real child services starting, a required child dying, the parent returning
non-zero, and the resulting child/port state.

## Environment

- OS: Windows AMD64, Node v24.15.0 (same as the PR's reported environment)
- Command: `node scripts/dev-with-automation.mjs --port 18101`
- Dependencies: committed lockfile (`node_modules` pre-installed); uvx 0.11.8
  (agent-server + automation packages cached after first run)

## Run 1 — full stack startup (baseline, before the kill)

All four services started as real child processes of the launcher
(`node scripts/dev-with-automation.mjs --port 18100`, launcher PID 43052):

```
launcher node.exe (PID 43052)
  uvx (PID 41036)
    uv (PID 9840)
      agent-server (PID 30812 -> 45612 -> 43036)   # uvicorn on 127.0.0.1:18000
  uvx (PID 36848)
    uv (PID 33308)
      uvicorn (PID 32688 -> 33316 -> 34276)        # automation on 127.0.0.1:18001
  cmd (PID 6240) -> node chain -> node (PID 12296)  # vite on 127.0.0.1:3001
  node (PID 41272)                                   # ingress on 127.0.0.1:18100
```

Listening ports at steady state:

```
127.0.0.1:3001   LISTENING  (vite)
127.0.0.1:18000 LISTENING  (agent-server, "Application startup complete")
127.0.0.1:18001 LISTENING  (automation backend)
127.0.0.1:18100 LISTENING  (ingress proxy, the advertised URL)
```

## Run 2 — required child dies, parent exit non-zero (the fix under test)

1. `node scripts/dev-with-automation.mjs --port 18101` (launcher PID 1946).
2. All four services came up again (vite :3001, agent-server :18000,
   automation :18001, ingress :18101).
3. Killed the required **vite** child (PID 37456, the listener on :3001) —
   the exact EADDRINUSE-class failure from issue #16927, forced directly:

   ```
   taskkill //PID 37456 //T //F
   → 成功: 已终止 PID 37456 (及其子进程 46660)的进程。
   ```

4. Launcher reaction (verbatim from `launcher-run2.log`):

   ```
   [vite] Exited with code 1
   ✗ Required service "vite" exited with code 1; stopping the stack.
   Shutting down...
   [agent-server] Stopping...
   [automation] Stopping...
   [ingress] Stopping...
   ```

5. Parent exit status: **`launcher exit code: 1`** (captured via `wait $!`).

6. Post-mortem child/port state:

   - Launcher process: gone.
   - Ports 3001 / 18000 / 18001 / 18101: all released (`netstat` clean).
   - No orphaned uvicorn/vite children left holding the ports.

## Interpretation

- Before the fix: the launcher logged `Exited with code 1` for the dying
  frontend and carried on — ingress bound the advertised port and the healthy
  banner printed while `localhost:8000` served a blank page.
- With the fix: the required child's death flips the launcher into
  `failService` → `shutdown({ exitCode: 1 })`: the remaining children are
  stopped, and the process exits non-zero, so a dead required service can
  never be reported as a healthy stack. The signal path (`code=null`,
  `signal!=null`) is covered by the same `failed` predicate.

## Artifacts

- `launcher-run1.log` — full startup log of run 1
- `launcher-run2.log` — startup + failure + shutdown log of run 2
- `process-tree-before-kill.txt` — process tree captured before the kill (run 1)
- `run2-exitcode.txt` — launcher PID, kill step, and captured exit code (run 2)
