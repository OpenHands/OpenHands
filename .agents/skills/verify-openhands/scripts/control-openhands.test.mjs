// Fast checks for control-openhands that need no browser or running stack.
// They run with the rest of the suite (`npm test`), or alone:
//   npx vitest run .agents/skills/verify-openhands
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "vitest";
import { buildIdentity } from "./lib/build-id.mjs";
import { BUILD_INPUTS } from "./lib/build-inputs.mjs";
import {
  editorOffset,
  launcherEnvFor,
  portsAfterRestart,
  runPorts,
} from "./lib/launcher-env.mjs";
import { redactStorage, redactStorageValue } from "./lib/redact-storage.mjs";
import { routePattern } from "./lib/route-pattern.mjs";
import { buildLocator, parseRole, toCss } from "./lib/selectors.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const cli = join(here, "control-openhands.mjs");

// A fake Playwright scope that records the locator chain it was asked for.
function recorder(path = []) {
  const step =
    (name) =>
    (...args) =>
      recorder([...path, [name, ...args]]);
  return {
    path,
    getByTestId: step("getByTestId"),
    getByRole: step("getByRole"),
    getByText: step("getByText"),
    getByLabel: step("getByLabel"),
    getByPlaceholder: step("getByPlaceholder"),
    getByTitle: step("getByTitle"),
    getByAltText: step("getByAltText"),
    nth: step("nth"),
    filter: step("filter"),
    locator: step("locator"),
  };
}

function run(args, env = {}) {
  const result = spawnSync(process.execPath, [cli, ...args], {
    encoding: "utf8",
    env: {
      ...process.env,
      OH_VERIFY_HOME: mkdtempSync(join(tmpdir(), "cov-")),
      OH_VERIFY_RUN: "",
      ...env,
    },
  });
  let json;
  try {
    json = JSON.parse(result.stdout);
  } catch {
    json = undefined;
  }
  return { ...result, json };
}

test("scoped testid chain builds nested locators", () => {
  const loc = buildLocator(
    recorder(),
    "testid=add-secret-form >> testid=submit-button",
  );
  assert.deepEqual(loc.path, [
    ["getByTestId", "add-secret-form"],
    ["getByTestId", "submit-button"],
  ]);
});

test("role selectors carry name, exactness and state options", () => {
  assert.deepEqual(parseRole('button[name="Save"][exact]'), {
    role: "button",
    options: { name: "Save", exact: true },
  });
  assert.deepEqual(parseRole("checkbox[checked=false]").options, {
    checked: false,
  });
  const regex = parseRole('link[name="/^Docs/i"]').options.name;
  assert.ok(regex instanceof RegExp && regex.test("docs page"));
  assert.ok(
    parseRole('option[name^="deepseek-pro"]').options.name.test(
      "deepseek-pro (default)",
    ),
  );
  assert.ok(
    !parseRole('option[name^="pro"]').options.name.test("deepseek-pro"),
  );
  assert.ok(
    parseRole('button[name*="SAVE"]').options.name.test("Save changes"),
  );
  assert.throws(
    () => parseRole("button[colour=red]"),
    /Unsupported role attribute/,
  );
});

test("testid segments accept attribute filters", () => {
  const loc = buildLocator(
    recorder(),
    'testid=onboarding-modal[data-current-step="1"]',
  );
  assert.deepEqual(loc.path, [
    ["locator", '[data-testid="onboarding-modal"][data-current-step="1"]'],
  ]);
});

test("quoted text is exact, bare text is partial, other segments pass through", () => {
  const loc = buildLocator(
    recorder(),
    'text="Secrets" >> text=Sec >> has-text=QA_ >> nth=1 >> visible >> input[type=file]',
  );
  assert.deepEqual(loc.path, [
    ["getByText", "Secrets", { exact: true }],
    ["getByText", "Sec", { exact: false }],
    ["filter", { hasText: "QA_" }],
    ["nth", 1],
    ["filter", { visible: true }],
    ["locator", "input[type=file]"],
  ]);
});

test("--help lists every command family with examples", () => {
  const { status, stdout } = run(["--help"]);
  assert.equal(status, 0);
  for (const word of [
    "launch",
    "doctor",
    "browser",
    "conversation",
    "evidence",
    "map",
    "Examples:",
  ]) {
    assert.match(stdout, new RegExp(word));
  }
  assert.match(run(["browser", "--help"]).stdout, /testids/);
});

test("usage errors exit 2 with a hint; a missing run exits 3", () => {
  const unknown = run(["frobnicate"]);
  assert.equal(unknown.status, 2);
  assert.equal(unknown.json.ok, false);
  assert.ok(unknown.json.hint);
  const noRun = run(["doctor"]);
  assert.equal(noRun.status, 3);
  assert.match(noRun.json.hint, /control-openhands launch/);
});

test("keys are refused on argv before any request is made", () => {
  const dir = mkdtempSync(join(tmpdir(), "cov-run-"));
  mkdirSync(join(dir, "private"));
  writeFileSync(join(dir, "private", "session-key"), "x".repeat(64));
  writeFileSync(
    join(dir, "run.json"),
    JSON.stringify({
      baseUrl: "http://127.0.0.1:9",
      ports: { ingress: 9 },
      launcherPgid: 0,
    }),
  );
  const result = run(
    ["llm", "set", "--profile", "x", "--model", "m", "--api-key", "secret"],
    { OH_VERIFY_RUN: dir },
  );
  assert.equal(result.status, 2);
  assert.match(result.json.error, /Do not pass keys on the command line/);
  assert.doesNotMatch(result.stdout, /secret"/);
});

test("map routes reads the route registry with nested settings paths", () => {
  const { status, json } = run(["map", "routes"]);
  assert.equal(status, 0);
  const paths = json.routes.map((r) => r.path);
  for (const path of [
    "/",
    "/settings",
    "/settings/llm",
    "/settings/secrets",
    "/automations/:automationId",
    "/shared/conversations/:conversationId",
  ]) {
    assert.ok(paths.includes(path), `missing ${path}`);
  }
});

test("toCss keeps testid/CSS chains in-page and defers engines to Playwright", () => {
  assert.equal(
    toCss("testid=chat-pane-header >> testid=ellipsis-button"),
    '[data-testid="chat-pane-header"] [data-testid="ellipsis-button"]',
  );
  assert.equal(
    toCss('testid=switch[data-state="on"] >> span.label'),
    '[data-testid="switch"][data-state="on"] span.label',
  );
  assert.equal(toCss('role=dialog >> role=button[name="Save"]'), null);
  assert.equal(toCss("testid=list >> nth=0"), null);
});

test("evidence retract is an evidence verb, not a conversation verb", () => {
  const res = spawnSync(process.execPath, [cli, "help", "evidence"], {
    encoding: "utf8",
  });
  assert.match(res.stdout, /evidence retract --feature ID/);
  const conv = spawnSync(process.execPath, [cli, "help", "conversation"], {
    encoding: "utf8",
  });
  assert.doesNotMatch(conv.stdout, /retract/);
  assert.match(conv.stdout, /--workspace PATH/);
});

test("browser help documents the input verbs agents asked for", () => {
  const res = spawnSync(process.execPath, [cli, "help", "browser"], {
    encoding: "utf8",
  });
  for (const verb of [
    "upload-via",
    "drop-files",
    "paste",
    "drag",
    "choose",
    "clipboard",
    "--expect-new-url",
    "--history",
  ])
    assert.match(res.stdout, new RegExp(verb.replace(/[-]/g, "\\-")));
});

test("has-text and visible need a previous segment", () => {
  assert.throws(
    () => buildLocator({ locator: () => ({}) }, "has-text=QA_x"),
    /narrows the previous segment/,
  );
});

test("global flags may come before the command", () => {
  const res = spawnSync(
    process.execPath,
    [cli, "--run", "/nonexistent-run", "status"],
    { encoding: "utf8" },
  );
  assert.doesNotMatch(res.stdout + res.stderr, /Unknown command/);
});

test("launch pins backend versions by flag and fixtures cover tarballs", () => {
  const launch = spawnSync(process.execPath, [cli, "help", "launch"], {
    encoding: "utf8",
  });
  assert.match(launch.stdout, /--sdk-version V/);
  assert.match(launch.stdout, /--automation-version V/);
  assert.match(launch.stdout, /not forwarded/);
  const fixture = spawnSync(process.execPath, [cli, "help", "fixture"], {
    encoding: "utf8",
  });
  assert.match(fixture.stdout, /fixture tarball/);
});

test("restart takes machine settings from the current shell, run settings from the saved env", () => {
  const saved = {
    HTTPS_PROXY: "http://127.0.0.1:39647",
    PATH: "/old/bin",
    UV_INDEX_URL: "https://old.example/simple",
    OH_CANVAS_SAFE_BACKEND_PORT: "18831",
    HOME: "/run/private/home",
  };
  const current = { HTTPS_PROXY: "http://127.0.0.1:43131", PATH: "/new/bin" };
  const env = launcherEnvFor(saved, current, [
    "HTTPS_PROXY",
    "PATH",
    "UV_INDEX_URL",
  ]);
  assert.equal(env.HTTPS_PROXY, "http://127.0.0.1:43131");
  assert.equal(env.PATH, "/new/bin");
  assert.equal(env.UV_INDEX_URL, undefined);
  assert.equal(env.OH_CANVAS_SAFE_BACKEND_PORT, "18831");
  assert.equal(env.HOME, "/run/private/home");
});

test("route patterns find parameterized paths the way the map writes them", () => {
  const conversation = routePattern("/conversations/:conversationId");
  assert.ok(conversation.test("open `/conversations/<id>` first"));
  assert.ok(conversation.test("`/conversations/abc-123`"));
  assert.ok(conversation.test("`/conversations/:conversationId`"));
  assert.ok(!conversation.test("`/conversations`"));
  const optional = routePattern("/settings/:tab?");
  assert.ok(optional.test("`/settings`") && optional.test("`/settings/app`"));
  assert.ok(
    routePattern("/extensions/:extensionName/*").test(
      "`/extensions/demo-page/hello`",
    ),
  );
});

test("map coverage maps parameterized routes by path, not only by file", () => {
  const res = spawnSync(process.execPath, [cli, "map", "coverage"], {
    encoding: "utf8",
  });
  const out = JSON.parse(res.stdout);
  for (const path of [
    "/conversations/:conversationId",
    "/automations/:automationId",
  ])
    assert.ok(!out.routes.fileOnly.includes(path), path);
  assert.deepEqual(out.routes.unmapped, []);
});

test("storage values hide secrets stored as plain strings and inside JSON", () => {
  const sidebar = JSON.stringify({ collapsed: true, width: 280, tabs: ["a"] });
  const out = redactStorage({
    "openhands-transcription-api-key": "sk-dummy-123",
    "openhands-session-key": JSON.stringify("abc"),
    "openhands-backends": JSON.stringify([
      { name: "Local", apiKey: "k-1", host: "http://127.0.0.1:1" },
    ]),
    "qa-profile": JSON.stringify({
      name: "Local",
      apiKey: { value: "k-2", rotated: false },
      tokens: ["k-3"],
      sessionKeys: { "k-4": { label: "laptop" } },
      password: 123456,
      refreshTokens: [],
      authenticated: true,
      connectionRevision: 3,
    }),
    // The zustand persist shape: the secret sits two objects deep.
    "qa-persisted": JSON.stringify({
      state: { apiKey: "k-5", authenticated: true },
      version: 0,
    }),
    "qa-sidebar": sidebar,
    "qa-json-string": JSON.stringify("local"),
    "qa-number": "42",
    "openhands-onboarded": "1",
    "openhands-theme": "light-plus",
  });
  assert.equal(out["openhands-transcription-api-key"], "<redacted>");
  assert.equal(out["openhands-session-key"], "<redacted>");
  assert.deepEqual(JSON.parse(out["openhands-backends"]), [
    { name: "Local", apiKey: "<redacted>", host: "http://127.0.0.1:1" },
  ]);
  // A secret-named field holding a number or a non-empty string, array or
  // object is hidden whole, property names included; empty values and
  // booleans next to it stay.
  assert.deepEqual(JSON.parse(out["qa-profile"]), {
    name: "Local",
    apiKey: "<redacted>",
    tokens: "<redacted>",
    sessionKeys: "<redacted>",
    password: "<redacted>",
    refreshTokens: [],
    authenticated: true,
    connectionRevision: 3,
  });
  assert.deepEqual(JSON.parse(out["qa-persisted"]), {
    state: { apiKey: "<redacted>", authenticated: true },
    version: 0,
  });
  assert.equal(out["qa-sidebar"], sidebar);
  assert.equal(out["qa-json-string"], JSON.stringify("local"));
  assert.equal(out["qa-number"], "42");
  assert.equal(out["openhands-onboarded"], "1");
  assert.equal(out["openhands-theme"], "light-plus");
  // Under a secret-looking storage key the whole value is hidden, whatever
  // its shape; only an empty value stays as it is.
  for (const value of [
    JSON.stringify(["sk-dummy-123"]),
    JSON.stringify({ v: "sk-dummy-123" }),
    "12345",
  ])
    assert.equal(
      redactStorageValue("openhands-transcription-api-key", value),
      "<redacted>",
    );
  assert.equal(redactStorageValue("openhands-transcription-api-key", ""), "");
  // Storage keys and JSON fields share one pattern.
  for (const key of ["openhands-llm-key", "qa-auth", "credentials", "passwd"])
    assert.equal(redactStorageValue(key, "sk-dummy-123"), "<redacted>");
});

test("launch refuses a run id that already exists or escapes the run home", () => {
  const home = mkdtempSync(join(tmpdir(), "runid-"));
  mkdirSync(join(home, "taken"));
  writeFileSync(join(home, "taken", "run.json"), "{}\n");
  const taken = run(["launch", "--new", "--run-id", "taken"], {
    OH_VERIFY_HOME: home,
  });
  assert.equal(taken.status, 2, taken.stderr);
  assert.match(taken.json.error, /already exists/);
  assert.equal(readFileSync(join(home, "taken", "run.json"), "utf8"), "{}\n");
  const escaping = run(["launch", "--new", "--run-id", "../outside"], {
    OH_VERIFY_HOME: home,
  });
  assert.equal(escaping.status, 2);
  assert.ok(!existsSync(join(home, "..", "outside")));
});

test("build inputs cover every file the frontend imports from outside src", () => {
  const repo = resolve(here, "../../../..");
  const covered = (path) =>
    BUILD_INPUTS.some(
      (input) => path === input || path.startsWith(`${input}/`),
    );
  const missing = new Set();
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (
        /\.(ts|tsx)$/.test(entry.name) &&
        !/\.test\.tsx?$/.test(entry.name)
      ) {
        const source = readFileSync(path, "utf8");
        for (const [, spec] of source.matchAll(
          /from\s+["'](\.{1,2}\/[^"']+)["']/g,
        )) {
          const target = relative(repo, resolve(dirname(path), spec));
          if (!covered(target)) missing.add(target);
        }
      }
    }
  };
  walk(join(repo, "src"));
  assert.deepEqual([...missing], []);
});

test("build identity changes when a new untracked source file appears", () => {
  const repo = mkdtempSync(join(tmpdir(), "buildid-"));
  const git = (...args) =>
    spawnSync("git", args, { cwd: repo, encoding: "utf8" });
  git("init", "-q");
  mkdirSync(join(repo, "src"));
  writeFileSync(join(repo, "src", "app.ts"), "export const a = 1;\n");
  writeFileSync(join(repo, ".gitignore"), "src/generated.ts\n");
  git("add", "-A");
  git(
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@t",
    "-c",
    "commit.gpgsign=false",
    "commit",
    "-qm",
    "init",
  );
  const clean = buildIdentity(repo, ["src"]);
  assert.doesNotMatch(clean, /-dirty$/);
  writeFileSync(join(repo, "src", "generated.ts"), "ignored\n");
  assert.equal(
    buildIdentity(repo, ["src"]),
    clean,
    "ignored files do not count",
  );
  writeFileSync(join(repo, "src", "not-found.tsx"), "export default 1;\n");
  const added = buildIdentity(repo, ["src"]);
  assert.notEqual(added, clean);
  assert.match(added, /-dirty$/);
  writeFileSync(join(repo, "src", "not-found.tsx"), "export default 2;\n");
  assert.notEqual(
    buildIdentity(repo, ["src"]),
    added,
    "edits to the new file count",
  );
});

// A checkout whose launcher decides the editor port with `body`, the way
// scripts/dev-with-automation.mjs buildConfig does.
function checkoutWithLauncher(body) {
  const repo = mkdtempSync(join(tmpdir(), "editor-"));
  mkdirSync(join(repo, "scripts"));
  writeFileSync(
    join(repo, "scripts", "dev-with-automation.mjs"),
    `export async function buildConfig(args, env) {
  const agentServerPort = Number(env.OH_CANVAS_SAFE_BACKEND_PORT);
  console.log("Checking ports...");
  ${body}
}\n`,
  );
  return repo;
}

test("the editor port is reserved only when the launcher serves VS Code", () => {
  const env = { PATH: process.env.PATH, LOCAL_BACKEND_API_KEY: "run-key" };
  // Since #17660: opt-in through OH_CANVAS_ENABLE_VSCODE=true.
  const optIn = checkoutWithLauncher(`return {
    agentServerPort,
    vscodePort: env.OH_CANVAS_ENABLE_VSCODE === "true" ? agentServerPort + 1000 : null,
  };`);
  const off = editorOffset(optIn, env);
  assert.deepEqual(off, { offset: null, source: "launcher" });
  assert.deepEqual(runPorts(18800, off.offset), {
    ingress: 18800,
    agentServer: 18801,
    automation: 18802,
    frontend: 18803,
  });
  const on = editorOffset(optIn, { ...env, OH_CANVAS_ENABLE_VSCODE: "true" });
  assert.deepEqual(on, { offset: 1000, source: "launcher" });
  assert.equal(runPorts(18800, on.offset).vscode, 19801);
  // Before #17660, and with #18048: always bundled. The probe never hands the
  // run's keys to the launcher it asks.
  const bundled = checkoutWithLauncher(`
  if (env.LOCAL_BACKEND_API_KEY === "run-key") throw new Error("leaked key");
  return { agentServerPort, vscodePort: agentServerPort + 1000 };`);
  assert.deepEqual(editorOffset(bundled, env), {
    offset: 1000,
    source: "launcher",
  });
  // A launcher that cannot be asked keeps the old reservation.
  const missing = editorOffset(mkdtempSync(join(tmpdir(), "editor-")), env);
  assert.equal(missing.source, "fallback");
  assert.equal(missing.offset, 1000);
  assert.match(missing.reason, /Cannot find module|ERR_MODULE_NOT_FOUND/);
});

test("this checkout's launcher answers the editor probe, and --vscode opts in", () => {
  const repo = resolve(here, "../../../..");
  const env = { PATH: process.env.PATH };
  // OH_CANVAS_ENABLE_VSCODE=true serves the editor on main and with #18048.
  assert.deepEqual(
    editorOffset(repo, { ...env, OH_CANVAS_ENABLE_VSCODE: "true" }),
    { offset: 1000, source: "launcher" },
  );
  // Without it: off on main (#17660), on again with #18048.
  const plain = editorOffset(repo, env);
  assert.equal(plain.source, "launcher", plain.reason);
  assert.ok([null, 1000].includes(plain.offset), String(plain.offset));
  const help = spawnSync(process.execPath, [cli, "help", "launch"], {
    encoding: "utf8",
  });
  assert.match(help.stdout, /--vscode/);
  assert.match(help.stdout, /OH_CANVAS_ENABLE_VSCODE/);
});

test("a restart follows the launcher's editor answer, or keeps its ports", () => {
  const without = runPorts(18800, null);
  const withEditor = runPorts(18800, 1000);
  const launcher = (offset) => ({ offset, source: "launcher" });
  // Unchanged answers change nothing and say nothing.
  assert.deepEqual(portsAfterRestart(without, launcher(null)), {
    ports: without,
  });
  assert.deepEqual(portsAfterRestart(withEditor, launcher(1000)), {
    ports: withEditor,
  });
  // The checkout started serving VS Code (say #18048 was pulled): the new
  // editor port is reported, for the caller to check before stopping.
  const gained = portsAfterRestart(without, launcher(1000));
  assert.deepEqual(gained.ports, withEditor);
  assert.equal(gained.added, 19801);
  assert.match(
    gained.warning,
    /now serves VS Code: added ports.vscode \(19801\)/,
  );
  // It stopped serving it: stop no longer waits on that port.
  const lost = portsAfterRestart(withEditor, launcher(null));
  assert.deepEqual(lost.ports, without);
  assert.equal(lost.added, undefined);
  assert.match(
    lost.warning,
    /no longer serves VS Code: dropped ports.vscode \(19801\)/,
  );
  // A launcher that cannot be asked (say mid-rebase) changes nothing.
  const unknown = portsAfterRestart(without, {
    offset: 1000,
    source: "fallback",
    reason: "Cannot find module",
  });
  assert.deepEqual(unknown.ports, without);
  assert.equal(unknown.added, undefined);
  assert.match(unknown.warning, /Cannot find module.*keeping this run's ports/);
});

test("restart refuses, before stopping anything, when a newly served editor port is busy", async () => {
  const listener = createServer();
  await new Promise((done) => listener.listen(0, "127.0.0.1", done));
  const busy = listener.address().port;
  const base = busy - 1001;
  try {
    // A run launched without the editor whose launcher now serves it (here
    // through the opt-in; with #18048, always). No keys are saved, so even a
    // restart that failed to refuse could not start anything.
    const dir = mkdtempSync(join(tmpdir(), "cov-run-"));
    mkdirSync(join(dir, "private"));
    writeFileSync(
      join(dir, "private", "launcher-env.json"),
      JSON.stringify({ OH_CANVAS_ENABLE_VSCODE: "true" }),
    );
    const before = JSON.stringify({
      baseUrl: `http://127.0.0.1:${base}`,
      ports: runPorts(base, null),
      launcherArgs: ["-e", "process.exit(1)"],
      launcherPgid: 0,
    });
    writeFileSync(join(dir, "run.json"), before);
    const result = run(["restart"], { OH_VERIFY_RUN: dir });
    assert.equal(result.status, 3, result.stdout + result.stderr);
    assert.match(
      result.json.error,
      new RegExp(`Port ${busy} \\(vscode\\) is busy`),
    );
    assert.match(result.json.hint, /still up/);
    assert.equal(readFileSync(join(dir, "run.json"), "utf8"), before);
  } finally {
    await new Promise((done) => listener.close(done));
  }
});
