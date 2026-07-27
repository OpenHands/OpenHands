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
 * Add spawn options needed for process-tree cleanup.
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
