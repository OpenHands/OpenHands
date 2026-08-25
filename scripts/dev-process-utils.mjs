import { spawnSync } from "node:child_process";
import process from "node:process";

/**
 * Return true while Node still considers the child process active.
 *
 * Do not use ChildProcess#killed for cleanup decisions. In Node, `killed`
 * only means a signal was sent successfully; it does not mean the process has
 * exited. That distinction matters for dev launchers because uvx/npm
 * wrappers can receive SIGTERM while their long-running child process keeps
 * serving on the original port.
 */
export function isProcessRunning(proc) {
  return proc.exitCode === null && proc.signalCode === null;
}

/**
 * Add spawn options needed for safe service launches and process-tree cleanup.
 *
 * Arguments must bypass shell parsing so values such as version constraints
 * containing `<` are forwarded literally. Callers that need shell behavior
 * must invoke the shell explicitly as the command.
 *
 * On POSIX, `detached: true` makes the spawned service the leader of a new
 * process group. Later we can signal `-pid` to terminate that whole group,
 * including wrapper chains like:
 *
 *   launcher -> uvx -> python agent-server
 *   launcher -> npm -> sh -> Vite
 *
 * Windows does not support POSIX process groups, so callers fall back to
 * signaling the direct child process there.
 */
export function getProcessTreeSpawnOptions(options = {}) {
  return {
    ...options,
    shell: false,
    detached: process.platform !== "win32",
  };
}

/**
 * Resolve a service command to a directly spawnable target on Windows.
 *
 * Services spawn without a shell so argument values reach the child verbatim.
 * Spawning `uvx` via cmd.exe instead makes it parse the args: a constraint like
 * `agent-client-protocol<0.11` is read as `<` input redirection and the spawn
 * dies with "The system cannot find the file specified." Resolving to an
 * absolute path lets callers spawn it shell-free.
 *
 * Returns `command` unchanged off Windows, when already a path, or if the lookup
 * fails.
 */
export function resolveWindowsCommand(
  command,
  platform = process.platform,
  lookup = whereCommandLookup,
) {
  if (platform !== "win32") {
    return command;
  }
  if (command.includes("/") || command.includes("\\")) {
    return command;
  }
  return lookup(command) || command;
}

function whereCommandLookup(command) {
  const result = spawnSync("where.exe", [command], { encoding: "utf8" });
  if (result.status !== 0 || !result.stdout) {
    return null;
  }
  return result.stdout.split(/\r?\n/).find(Boolean)?.trim() || null;
}

/**
 * Signal the whole spawned service tree when possible.
 *
 * POSIX `process.kill(-pid, signal)` targets the process group whose id is
 * `pid`; this only works because services are spawned with
 * `getProcessTreeSpawnOptions()`. Without the negative pid, shutdown would
 * often stop only the wrapper process and leave the actual server child
 * listening on its port.
 */
export function signalProcessTree(proc, signal) {
  if (!isProcessRunning(proc)) {
    return false;
  }

  try {
    if (process.platform === "win32" && proc.pid) {
      killWindowsProcessTree(proc, signal);
    } else if (!proc.pid) {
      proc.kill(signal);
    } else {
      process.kill(-proc.pid, signal);
    }
    return true;
  } catch (err) {
    if (err?.code === "ESRCH") {
      return false;
    }
    throw err;
  }
}

/**
 * Windows has no POSIX process groups: ChildProcess#kill reaches only the
 * direct child (e.g. the uvx wrapper), leaving grandchildren — the actual
 * python agent-server holding its port — running. `taskkill /t` walks the
 * child tree instead. Windows also has no graceful tree signal (taskkill
 * without /f posts WM_CLOSE, which console processes ignore), so SIGTERM and
 * SIGKILL both map to the same forceful /f kill; callers' delayed SIGKILL
 * pass skips already-exited trees via isProcessRunning, so the repeat is a
 * no-op. A non-zero taskkill exit just means the tree already exited — only
 * a failure to spawn taskkill itself falls back to the direct kill.
 */
function killWindowsProcessTree(proc, signal) {
  const result = spawnSync(
    "taskkill",
    ["/pid", String(proc.pid), "/t", "/f"],
    // windowsHide avoids a console window flash when invoked from the
    // packaged (GUI) Electron process.
    { stdio: "ignore", windowsHide: true },
  );
  if (result.error) {
    proc.kill(signal);
  }
}

export function createShutdownHookRegistry(onError) {
  const hooks = new Set();

  return {
    add(hook) {
      hooks.add(hook);
      return () => hooks.delete(hook);
    },

    run() {
      for (const hook of hooks) {
        try {
          hook();
        } catch (err) {
          onError?.(err);
        }
      }
    },
  };
}

/**
 * Watch a spawned automation backend for a stale SQLite migration failure
 * and recover once by deleting the stale DB and restarting.
 *
 * The failure happens when openhands-automation is updated to a version
 * with a restructured Alembic revision chain: the old DB references a
 * revision the new chain doesn't know about, and the backend exits with
 * Alembic's "Can't locate revision identified by" error.
 *
 * Both dev launchers (dev-static and dev-with-automation) need exactly
 * this behaviour; keeping it here means they cannot drift apart.
 *
 * Recovery is single-shot by design. If the restart also fails with a
 * migration error — broken upstream package, read-only filesystem,
 * changed error wording — we log loudly and give up rather than loop.
 * An unbounded respawn cycle would wedge `npm run dev` and mask the real
 * breakage from the developer.
 *
 * Params are injectable so tests can drive the whole lifecycle with fake
 * streams and timers.
 */
export function watchAutomationMigration(
  proc,
  { dbPath, onRecover, log, pattern = "Can't locate revision identified by" },
) {
  let detected = false;
  // Single-shot latch, scoped to THIS watcher instance. One recovery
  // attempt per launcher run: if the restarted backend fails the same way
  // again we log loudly and give up rather than respawn forever.
  let attempted = false;

  const checkForMigrationError = (data) => {
    if (!detected && data.toString().includes(pattern)) {
      detected = true;
    }
  };

  proc.stdout?.on("data", checkForMigrationError);
  proc.stderr?.on("data", checkForMigrationError);

  proc.on("exit", (code) => {
    if (!detected) {
      return;
    }

    // Exit code 3 is how the automation backend currently surfaces an
    // unrecoverable startup error. We still require the log pattern so a
    // coincidental exit 3 from another cause never deletes the DB.
    if (code !== 3) {
      return;
    }

    // Single-shot latch: one recovery attempt per launcher run, ever.
    if (attempted) {
      log?.("Migration error again after recovery — giving up.", "error");
      return;
    }
    attempted = true;

    log?.(`Migration error detected — removing stale DB at ${dbPath}...`, "warn");
    try {
      onRecover();
      log?.("Deleted stale automations.db, restarting...", "ok");
    } catch (err) {
      log?.(`Failed to remove stale automations.db: ${err.message}`, "error");
    }
  });
}
