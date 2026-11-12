import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "vitest";
import { buildIdentity } from "./lib/build-id.mjs";

const here = dirname(fileURLToPath(import.meta.url));

// Exercise the actual CLI and npm command in a disposable Git checkout. The
// fixture launcher/build prove harness lifecycle and identity bookkeeping only;
// they do not stand in for a product build or an OpenHands UI verification.
async function fixture() {
  const root = mkdtempSync(join(tmpdir(), "ohv-rebuild-"));
  const repo = join(root, "repo");
  const home = join(root, "runs");
  const scripts = join(repo, ".agents/skills/verify-openhands/scripts");
  for (const path of [
    scripts,
    home,
    join(repo, "src"),
    join(repo, "build"),
    join(repo, "node_modules"),
    join(repo, "bin"),
    join(repo, "scripts"),
  ])
    mkdirSync(path, { recursive: true });
  cpSync(join(here, "lib"), join(scripts, "lib"), { recursive: true });
  cpSync(
    join(here, process.env.OH_VERIFY_TEST_CLI ?? "control-openhands.mjs"),
    join(scripts, "control-openhands.mjs"),
  );
  const map = join(scripts, "../references/feature-map");
  mkdirSync(map, { recursive: true });
  writeFileSync(
    join(map, "F29-fixture.md"),
    "# F29 fixture\n\n## Sub-features\n\n- `F29.entry`: fixture only.\n",
  );
  writeFileSync(
    join(repo, ".gitignore"),
    "node_modules/\nbuild/\n.agents/\nqa-*.txt\n",
  );
  writeFileSync(join(repo, "src/main.js"), "before\n");
  writeFileSync(
    join(repo, "package.json"),
    JSON.stringify({
      private: true,
      scripts: { "build:app": "node qa-build.mjs" },
    }),
  );
  writeFileSync(
    join(repo, "scripts/dev-with-automation.mjs"),
    `
export function buildConfig(_args, env) {
  return { agentServerPort: Number(env.OH_CANVAS_SAFE_BACKEND_PORT), vscodePort: null };
}
`,
  );
  writeFileSync(
    join(repo, "bin/agent-canvas.mjs"),
    `
import { createServer } from "node:http";
import { appendFileSync } from "node:fs";
appendFileSync(process.env.QA_LAUNCH_COUNTER, "started\\n");
const servers = Array.from({ length: 4 }, (_, offset) => {
  const server = createServer((_req, res) => {
    res.setHeader("content-type", "application/json");
    res.end("{}");
  });
  server.listen(Number(process.env.QA_PORT) + offset, "127.0.0.1");
  return server;
});
process.on("SIGTERM", async () => {
  await Promise.all(servers.map(server => new Promise(resolve => server.close(resolve))));
  process.exit(0);
});
`,
  );
  writeFileSync(
    join(repo, "qa-build.mjs"),
    `
import { createConnection } from "node:net";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
const port = Number(readFileSync("qa-port.txt", "utf8"));
const stillServing = await new Promise((resolve) => {
  const socket = createConnection({ port, host: "127.0.0.1" });
  socket.once("connect", () => { socket.destroy(); resolve(true); });
  socket.once("error", () => resolve(false));
});
if (stillServing) process.exit(91);
const mode = existsSync("qa-mode.txt") ? readFileSync("qa-mode.txt", "utf8").trim() : "ok";
mkdirSync("build", { recursive: true });
writeFileSync("build/index.html", readFileSync("src/main.js", "utf8"));
if (mode === "fail") process.exit(17);
if (mode === "drift") writeFileSync("src/main.js", "changed during build\\n");
`,
  );
  const git = (...args) => {
    const result = spawnSync("git", args, { cwd: repo, encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  git("init", "-q");
  git("add", ".");
  const commit = (message) => {
    git("add", "src");
    git(
      "-c",
      "user.name=Harness test",
      "-c",
      "user.email=qa@example.invalid",
      "-c",
      "commit.gpgsign=false",
      "commit",
      "-qm",
      message,
    );
    return git("rev-parse", "HEAD");
  };
  const revision = commit("baseline");
  const buildId = buildIdentity(repo);
  writeFileSync(join(repo, "build/index.html"), "before\n");
  writeFileSync(join(repo, "build/verify-revision.txt"), `${buildId}\n`);
  const runDirs = [];
  const children = [];
  const makeRun = async (name) => {
    const dir = join(home, name);
    runDirs.push(dir);
    for (const part of ["private", "private/home", "evidence", "workspace"])
      mkdirSync(join(dir, part), { recursive: true });
    let port;
    for (let attempt = 0; attempt < 20 && port === undefined; attempt += 1) {
      const reservations = [];
      try {
        for (let offset = 0; offset < 4; offset += 1) {
          const listener = createServer();
          reservations.push(listener);
          const requested = offset
            ? reservations[0].address().port + offset
            : 0;
          await new Promise((resolve, reject) => {
            listener.once("error", reject);
            listener.listen(requested, "127.0.0.1", resolve);
          });
        }
        port = reservations[0].address().port;
      } catch {
      } finally {
        await Promise.all(
          reservations.map(
            (listener) => new Promise((resolve) => listener.close(resolve)),
          ),
        );
      }
    }
    assert.ok(port, "fixture needs an unused four-port block");
    const saved = {
      QA_PORT: String(port),
      QA_LAUNCH_COUNTER: join(dir, "private/launches.txt"),
      HOME: join(dir, "private/home"),
    };
    writeFileSync(
      join(dir, "private/launcher-env.json"),
      JSON.stringify(saved),
    );
    writeFileSync(join(dir, "private/session-key"), "fixture-session-key");
    writeFileSync(
      join(dir, "private/encryption-key"),
      "fixture-encryption-key",
    );
    writeFileSync(join(dir, "private/state-marker"), "preserve-state");
    writeFileSync(
      join(dir, "evidence/ledger.jsonl"),
      `${JSON.stringify({ feature: "F29.entry", entry: "baseline", revision, buildId, result: "fail" })}\n`,
    );
    const args = [join(repo, "bin/agent-canvas.mjs")];
    const child = spawn(process.execPath, args, {
      cwd: repo,
      env: { ...process.env, ...saved },
      detached: true,
      stdio: "ignore",
    });
    children.push(child);
    const run = {
      runId: name,
      repo,
      revision,
      buildId,
      startedAt: "baseline-time",
      baseUrl: `http://127.0.0.1:${port}`,
      ports: {
        ingress: port,
        agentServer: port + 1,
        automation: port + 2,
        frontend: port + 3,
      },
      launcherArgs: args,
      launcherPid: child.pid,
      launcherPgid: child.pid,
    };
    writeFileSync(join(dir, "run.json"), JSON.stringify(run));
    for (let i = 0; i < 100; i += 1) {
      try {
        if ((await fetch(run.baseUrl)).ok) return { dir, run };
      } catch {}
      await delay(20);
    }
    throw new Error("Fixture launcher did not listen");
  };
  const invokeCommand = (dir, command, ...args) =>
    new Promise((resolve, reject) => {
      const child = spawn(
        process.execPath,
        [
          join(scripts, "control-openhands.mjs"),
          command,
          ...args,
          "--run",
          dir,
        ],
        {
          cwd: repo,
          env: { ...process.env, OH_VERIFY_HOME: home, OH_VERIFY_RUN: "" },
        },
      );
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk;
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk;
      });
      child.on("error", reject);
      child.on("close", (status) => {
        try {
          resolve({ status, stdout, stderr, json: JSON.parse(stdout) });
        } catch (error) {
          reject(new Error(`${error.message}: ${stdout} ${stderr}`));
        }
      });
    });
  const invoke = (dir, ...args) => invokeCommand(dir, "restart", ...args);
  const readRun = (dir) =>
    JSON.parse(readFileSync(join(dir, "run.json"), "utf8"));
  const cleanup = async () => {
    for (const dir of runDirs) {
      try {
        process.kill(-readRun(dir).launcherPgid, "SIGTERM");
      } catch {}
    }
    for (const child of children) {
      try {
        process.kill(-child.pid, "SIGTERM");
      } catch {}
    }
    await delay(100);
    rmSync(root, { recursive: true, force: true });
  };
  return {
    repo,
    home,
    revision,
    buildId,
    makeRun,
    invoke,
    invokeCommand,
    readRun,
    commit,
    cleanup,
  };
}

test("restart --rebuild stops the owned server, preserves state and records actual old/new identities", async () => {
  const f = await fixture();
  try {
    const { dir, run } = await f.makeRun("owned");
    writeFileSync(join(f.repo, "qa-port.txt"), String(run.ports.ingress));
    const ledgerBefore = readFileSync(
      join(dir, "evidence/ledger.jsonl"),
      "utf8",
    );
    writeFileSync(join(f.repo, "src/main.js"), "after\n");
    const afterRevision = f.commit("fix");
    // Ordinary restart retains the existing build and identity.
    const ordinary = await f.invoke(dir, "--timeout", "10");
    assert.equal(ordinary.status, 0, ordinary.stdout + ordinary.stderr);
    assert.equal(f.readRun(dir).revision, f.revision);
    assert.equal(
      readFileSync(join(f.repo, "build/index.html"), "utf8"),
      "before\n",
    );
    const result = await f.invoke(dir, "--rebuild", "--timeout", "10");
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.equal(result.json.rebuiltFrontend, true);
    const after = f.readRun(dir);
    assert.equal(after.revision, afterRevision);
    assert.equal(after.buildId, buildIdentity(f.repo));
    assert.equal(
      readFileSync(join(f.repo, "build/verify-revision.txt"), "utf8").trim(),
      after.buildId,
    );
    assert.equal(
      readFileSync(join(f.repo, "build/index.html"), "utf8"),
      "after\n",
    );
    assert.equal(after.runId, run.runId);
    assert.equal(after.startedAt, run.startedAt);
    assert.deepEqual(after.ports, run.ports);
    assert.equal(
      readFileSync(join(dir, "private/state-marker"), "utf8"),
      "preserve-state",
    );
    assert.equal(
      readFileSync(join(dir, "private/session-key"), "utf8"),
      "fixture-session-key",
    );
    assert.equal(
      readFileSync(join(dir, "evidence/ledger.jsonl"), "utf8"),
      ledgerBefore,
    );
    const history = readFileSync(
      join(dir, "evidence/build-history.jsonl"),
      "utf8",
    )
      .trim()
      .split("\n")
      .map(JSON.parse);
    assert.deepEqual(
      history.map((row) => row.event),
      ["started", "succeeded"],
    );
    assert.deepEqual(history[1].previous, {
      revision: f.revision,
      buildId: f.buildId,
    });
    assert.deepEqual(history[1].current, {
      revision: afterRevision,
      buildId: after.buildId,
    });
    assert.deepEqual(after.buildHistory, history);
    const evidence = await f.invokeCommand(
      dir,
      "evidence",
      "add",
      "--feature",
      "F29.entry",
      "--entry",
      "fixture-after",
      "--result",
      "not-run",
    );
    assert.equal(evidence.status, 0, evidence.stdout + evidence.stderr);
    assert.equal(evidence.json.recorded.buildId, after.buildId);
    assert.equal(evidence.json.recorded.revision, afterRevision);
    const report = await f.invokeCommand(dir, "evidence", "report", "--json");
    assert.equal(report.status, 0, report.stdout + report.stderr);
    assert.deepEqual(report.json.buildHistory, history);
    assert.equal(report.json.rows[0].revision, f.revision);
    assert.equal(report.json.rows[1].revision, afterRevision);
  } finally {
    await f.cleanup();
  }
});

test("another live run using this checkout refuses rebuilding before either server is stopped", async () => {
  const f = await fixture();
  try {
    const first = await f.makeRun("owned");
    const other = await f.makeRun("other");
    const before = readFileSync(join(first.dir, "run.json"), "utf8");
    const result = await f.invoke(first.dir, "--rebuild");
    assert.equal(result.status, 3, result.stdout + result.stderr);
    assert.match(result.json.error, /Other live verification runs/);
    assert.deepEqual(result.json.conflicts, [other.dir]);
    assert.equal(readFileSync(join(first.dir, "run.json"), "utf8"), before);
    assert.equal((await fetch(first.run.baseUrl)).status, 200);
    assert.equal((await fetch(other.run.baseUrl)).status, 200);
    assert.equal(
      readFileSync(join(f.repo, "build/index.html"), "utf8"),
      "before\n",
    );
  } finally {
    await f.cleanup();
  }
});

test("failed rebuild retains identity/evidence and blocks plain restart until a successful retry", async () => {
  const f = await fixture();
  try {
    const { dir, run } = await f.makeRun("owned");
    writeFileSync(join(f.repo, "qa-port.txt"), String(run.ports.ingress));
    writeFileSync(join(f.repo, "qa-mode.txt"), "fail");
    const result = await f.invoke(dir, "--rebuild");
    assert.equal(result.status, 3, result.stdout + result.stderr);
    assert.match(result.json.error, /remains stopped.*exit 17/);
    assert.equal(f.readRun(dir).frontendRebuild.status, "failed");
    assert.equal(f.readRun(dir).buildId, f.buildId);
    assert.equal(f.readRun(dir).revision, f.revision);
    assert.equal(existsSync(join(f.repo, "build/verify-revision.txt")), false);
    await assert.rejects(fetch(run.baseUrl));
    assert.equal(
      readFileSync(join(dir, "private/launches.txt"), "utf8"),
      "started\n",
    );
    const ordinary = await f.invoke(dir);
    assert.equal(ordinary.status, 3);
    assert.match(ordinary.json.error, /plain restart refused/);
    writeFileSync(join(f.repo, "qa-mode.txt"), "ok");
    const retry = await f.invoke(dir, "--rebuild", "--timeout", "10");
    assert.equal(retry.status, 0, retry.stdout + retry.stderr);
    assert.deepEqual(
      f.readRun(dir).buildHistory.map((row) => row.event),
      ["started", "failed", "started", "succeeded"],
    );
    assert.equal(
      readFileSync(join(dir, "private/state-marker"), "utf8"),
      "preserve-state",
    );
    assert.equal(
      readFileSync(join(dir, "evidence/ledger.jsonl"), "utf8").trim(),
      JSON.stringify({
        feature: "F29.entry",
        entry: "baseline",
        revision: f.revision,
        buildId: f.buildId,
        result: "fail",
      }),
    );
  } finally {
    await f.cleanup();
  }
});

test("source changes during a successful build are rejected without stamping or relaunching", async () => {
  const f = await fixture();
  try {
    const { dir, run } = await f.makeRun("owned");
    writeFileSync(join(f.repo, "qa-port.txt"), String(run.ports.ingress));
    writeFileSync(join(f.repo, "qa-mode.txt"), "drift");
    const result = await f.invoke(dir, "--rebuild");
    assert.equal(result.status, 3, result.stdout + result.stderr);
    assert.match(result.json.error, /build inputs changed during the build/);
    assert.equal(f.readRun(dir).buildId, f.buildId);
    assert.equal(existsSync(join(f.repo, "build/verify-revision.txt")), false);
    await assert.rejects(fetch(run.baseUrl));
  } finally {
    await f.cleanup();
  }
});

test("invalid rebuild option and interrupted prior build refuse restart without mutation", async () => {
  const f = await fixture();
  try {
    const { dir, run } = await f.makeRun("owned");
    const invalid = await f.invoke(dir, "--rebuild=false");
    assert.equal(invalid.status, 2);
    assert.equal((await fetch(run.baseUrl)).status, 200);
    const pending = {
      ...f.readRun(dir),
      frontendRebuild: { status: "building" },
    };
    writeFileSync(join(dir, "run.json"), JSON.stringify(pending));
    const result = await f.invoke(dir);
    assert.equal(result.status, 3);
    assert.match(result.json.error, /did not complete/);
    assert.equal((await fetch(run.baseUrl)).status, 200);
    assert.deepEqual(f.readRun(dir), pending);
  } finally {
    await f.cleanup();
  }
});
