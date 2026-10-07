import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// The environment for (re)starting a run's launcher. Run-scoped values
// (ports, state paths, private HOME) come from the environment saved at
// launch. Machine-scoped values (PATH, proxy, CA bundles: the pass-through
// names) come from the current shell, because they can change between
// sessions: a restart that replayed a stale proxy port could no longer reach
// the model provider.
export function launcherEnvFor(saved, current, passNames) {
  const env = { ...saved };
  for (const name of passNames) {
    if (current[name] !== undefined) env[name] = current[name];
    else delete env[name];
  }
  return env;
}

// Whether a launch of this checkout with a given environment serves the
// bundled VS Code editor, and on which port. OpenHands/OpenHands#17660 made
// the editor opt-in (OH_CANVAS_ENABLE_VSCODE=true) and OpenHands/OpenHands#18048
// bundled it again, so neither is assumed: the checkout's own launcher is
// asked. bin/agent-canvas.mjs runs buildConfig from
// scripts/dev-with-automation.mjs, whose vscodePort is null when the editor is
// off.
//
// The question runs in a child process with a throwaway HOME and state
// directory, because importing the launcher's modules creates a log directory
// under the state directory (by default the operator's ~/.openhands).

// Where the launcher put the editor before #17660, and with
// OH_CANVAS_ENABLE_VSCODE=true since: agent-server port + 1000. Used only when
// the launcher cannot be asked, so a failed probe reserves as much as before.
export const FALLBACK_EDITOR_OFFSET = 1000;

const PROBE = `
import { createServer } from "node:net";
// stdout carries only the answer.
const answer = (value) => process.stdout.write(JSON.stringify(value));
console.log = console.info = (...args) => console.error(...args);
try {
  // Free ports for the launcher's own port check; the answer does not depend
  // on which ports they are.
  const servers = [];
  for (let i = 0; i < 4; i += 1)
    servers.push(await new Promise((resolve, reject) => {
      const server = createServer().once("error", reject);
      server.listen(0, "127.0.0.1", () => resolve(server));
    }));
  const [ingress, agentServer, automation, frontend] = servers.map((s) => s.address().port);
  await Promise.all(servers.map((s) => new Promise((resolve) => s.close(resolve))));
  const { buildConfig } = await import(process.argv[1]);
  const config = await buildConfig({ port: ingress }, {
    ...process.env,
    OH_CANVAS_SAFE_BACKEND_PORT: String(agentServer),
    OH_CANVAS_SAFE_AUTOMATION_PORT: String(automation),
    OH_CANVAS_SAFE_VITE_PORT: String(frontend),
  });
  answer({ agentServer: config.agentServerPort, vscode: config.vscodePort ?? null });
} catch (error) {
  answer({ error: String(error?.message ?? error).split("\\n")[0] });
}
`;

/**
 * Ask the checkout's launcher whether it serves the editor with `env` (the
 * environment the launcher will get), as an offset from the agent-server port.
 * @returns {{ offset: number | null, source: "launcher" | "fallback", reason?: string }}
 */
export function editorOffset(repoRoot, env) {
  const launcher = join(repoRoot, "scripts", "dev-with-automation.mjs");
  const home = mkdtempSync(join(tmpdir(), "ohv-editor-probe-"));
  try {
    const result = spawnSync(
      process.execPath,
      // argv[1] must stay a file URL: the launcher runs main() (a whole stack)
      // only when argv[1] is its plain path, so a URL imports it inertly.
      ["--input-type=module", "-e", PROBE, pathToFileURL(launcher).href],
      {
        cwd: repoRoot,
        encoding: "utf8",
        timeout: 30_000,
        env: {
          ...env,
          HOME: home,
          OH_CANVAS_SAFE_STATE_DIR: join(home, "state"),
          // Placeholders: the probe reads no key, and the real ones stay out.
          LOCAL_BACKEND_API_KEY: "editor-probe",
          OH_SECRET_KEY: "editor-probe",
        },
      },
    );
    let answer;
    try {
      answer = JSON.parse(result.stdout);
    } catch {
      // No answer at all: the launcher exited, or the probe could not start.
      throw new Error(
        result.error?.message ||
          `probe exited ${result.status ?? result.signal} without an answer`,
      );
    }
    if (answer.error) throw new Error(answer.error);
    if (answer.vscode === null) return { offset: null, source: "launcher" };
    if (Number.isInteger(answer.vscode) && Number.isInteger(answer.agentServer))
      return { offset: answer.vscode - answer.agentServer, source: "launcher" };
    throw new Error(`unexpected answer ${result.stdout}`);
  } catch (error) {
    return {
      offset: FALLBACK_EDITOR_OFFSET,
      source: "fallback",
      reason: String(error.message).slice(0, 300),
    };
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
}

/**
 * A run's ports from its block base: ingress P, agent-server P+1, automation
 * P+2, static frontend P+3, and the editor only when the launcher serves it.
 */
export function runPorts(base, offset) {
  return {
    ingress: base,
    agentServer: base + 1,
    automation: base + 2,
    frontend: base + 3,
    ...(offset === null ? {} : { vscode: base + 1 + offset }),
  };
}

/**
 * A restarted run's ports. The block never moves, but the checkout may have
 * changed since launch, so the editor port follows the launcher's new answer
 * (`editor`, from editorOffset). When the launcher cannot be asked, the run
 * keeps the ports it had. `added` is an editor port the run did not use
 * before: the caller checks that it is free.
 * @returns {{ ports: Record<string, number>, added?: number, warning?: string }}
 */
export function portsAfterRestart(previous, editor) {
  if (editor.source === "fallback")
    return {
      ports: previous,
      warning: `Could not ask scripts/dev-with-automation.mjs whether it serves VS Code (${editor.reason}); keeping this run's ports.`,
    };
  const ports = runPorts(previous.ingress, editor.offset);
  if (ports.vscode === previous.vscode) return { ports };
  if (ports.vscode === undefined)
    return {
      ports,
      warning: `The checkout's launcher no longer serves VS Code: dropped ports.vscode (${previous.vscode}).`,
    };
  return {
    ports,
    added: ports.vscode,
    warning: `The checkout's launcher now serves VS Code: added ports.vscode (${ports.vscode}).`,
  };
}
