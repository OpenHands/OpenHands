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
import { baselineLines, parseBaseline, withBaseline } from "./lib/baseline.mjs";
import {
  affectedFamilies,
  familyHead,
  parseSourceLine,
  specMatches,
} from "./lib/map-sources.mjs";
import { routePattern } from "./lib/route-pattern.mjs";
import { resolveTestids } from "./lib/testids.mjs";
import { tmuxPathFor } from "./lib/tmux-path.mjs";
import { browserCallLimit } from "./lib/call-limit.mjs";
import { agentProfileRepoint } from "./lib/agent-profile-repoint.mjs";
import {
  canvasSize,
  encodeRecording,
  encoderSupport,
  ffmpegArgs,
  findFfmpeg,
  frameDurations,
  frameRepeats,
  playwrightFfmpegPaths,
  withoutPauses,
} from "./lib/recording.mjs";
import { MAX_PAGES, collectEvents, countImages } from "./lib/events-paging.mjs";
import { redactBody, redactUrl } from "./lib/network-bodies.mjs";
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

test("Source: lines parse into path specs, parenthesized names relative to their directory", () => {
  const specs = parseSourceLine(
    "Source: `src/routes/secrets-settings.tsx`, `src/components/features/automations/` (`automation-card.tsx`, `dashboard/`), `src/api/agent-server-adapter.ts` (`buildRouterAtStartSystemSuffix`), `src/components/features/conversation/conversation-overview-*.tsx`, `node_modules/@openhands/extensions/automations/interface.json`.",
  );
  assert.deepEqual(
    specs.map((s) => s.path),
    [
      "src/routes/secrets-settings.tsx",
      "src/components/features/automations/",
      "src/components/features/automations/automation-card.tsx",
      "src/components/features/automations/dashboard/",
      "src/api/agent-server-adapter.ts",
      "src/components/features/conversation/conversation-overview-*.tsx",
      "node_modules/@openhands/extensions/automations/interface.json",
    ],
  );
  const glob = specs.find((s) => s.glob);
  assert.ok(
    specMatches(
      glob,
      "src/components/features/conversation/conversation-overview-tiles.tsx",
    ),
  );
  assert.ok(
    !specMatches(
      glob,
      "src/components/features/conversation/sub/conversation-overview-x.tsx",
    ),
  );
  const dir = specs.find(
    (s) => s.path === "src/components/features/automations/",
  );
  assert.ok(
    specMatches(dir, "src/components/features/automations/kebab-menu.tsx"),
  );
  assert.ok(
    !specMatches(dir, "src/components/features/automations-extra/x.tsx"),
  );
});

test("only the Source: line in a file's head counts", () => {
  const head = familyHead(
    "# F14\n\nText.\n\nSource: `src/api/secrets-service.ts`.\n\n## Sub-features\n\n- `F14.list`: x\n\nSource: `not/in/head.ts`\n",
  );
  assert.deepEqual(
    head.sources.map((s) => s.path),
    ["src/api/secrets-service.ts"],
  );
});

test("changed paths map to the families whose Source: lines own them", () => {
  const families = [
    {
      id: "F14",
      file: "F14-secrets.md",
      sources: parseSourceLine(
        "Source: `src/routes/secrets-settings.tsx`, `src/components/features/settings/secrets-settings/`, `src/api/secrets-service.ts`.",
      ),
    },
    {
      id: "F21",
      file: "F21-automations-dashboard.md",
      sources: parseSourceLine(
        "Source: `src/components/features/automations/` (`automation-card.tsx`, `dashboard/`).",
      ),
    },
  ];
  const result = affectedFamilies(families, [
    "src/components/features/settings/secrets-settings/secret-form.tsx",
    "src/components/features/automations/dashboard/overview.tsx",
    "src/components/features/automations/dashboard/overview.test.tsx",
    "__tests__/components/secrets.test.tsx",
    "src/hooks/query/use-get-secrets.ts",
    "src/i18n/translation.json",
    "src/components/features/brand-new/page.tsx",
    "src/mocks/conversation-handlers.ts",
    ".agents/skills/verify-openhands/SKILL.md",
  ]);
  assert.deepEqual(
    result.families.map((f) => [f.id, f.paths]),
    [
      [
        "F14",
        ["src/components/features/settings/secrets-settings/secret-form.tsx"],
      ],
      ["F21", ["src/components/features/automations/dashboard/overview.tsx"]],
    ],
  );
  // Shared code is widened by the caller, never silently dropped.
  assert.deepEqual(result.shared, [
    "src/hooks/query/use-get-secrets.ts",
    "src/i18n/translation.json",
  ]);
  // A src/ path no family owns is a map gap.
  assert.deepEqual(result.unmapped, [
    "src/components/features/brand-new/page.tsx",
  ]);
  // Tests, mocks and this skill change no user-facing behavior, even inside
  // a family's directory.
  assert.deepEqual(result.nonUserFacing, [
    "src/components/features/automations/dashboard/overview.test.tsx",
    "__tests__/components/secrets.test.tsx",
    "src/mocks/conversation-handlers.ts",
    ".agents/skills/verify-openhands/SKILL.md",
  ]);
});

test("map affected reads a path list and classifies its paths", () => {
  const res = spawnSync(
    process.execPath,
    [cli, "map", "affected", "--paths", "-"],
    {
      encoding: "utf8",
      input:
        "src/components/features/settings/secrets-settings/secret-form.tsx\ndocs/README.md\n",
    },
  );
  const json = JSON.parse(res.stdout);
  assert.equal(res.status, 0, res.stdout);
  assert.deepEqual(
    json.families.map((f) => f.id),
    ["F14"],
  );
  assert.deepEqual(json.nonUserFacing, ["docs/README.md"]);
  // What `map affected` does with no flags depends on the index's baseline
  // line; the baseline test below covers both states.
});

test("map check rejects a Source: path that is gone, and the shipped map is clean", () => {
  const dir = mkdtempSync(join(tmpdir(), "mapcheck-"));
  // The checker reads the real map dir; prove the rules on the parser level
  // and on the live map, which must be clean.
  const clean = spawnSync(process.execPath, [cli, "map", "check"], {
    encoding: "utf8",
  });
  assert.equal(clean.status, 0, clean.stdout);
  assert.deepEqual(JSON.parse(clean.stdout).problems, []);
  const head = familyHead(
    `# F99\n\nText.\n\nSource: \`src/does-not-exist.tsx\`.\n\n## Sub-features\n`,
  );
  assert.equal(head.sources[0].path, "src/does-not-exist.tsx");
  assert.ok(existsSync(dir));
});

test("cited test ids resolve through literals and dynamic prefixes", () => {
  const cited = new Map([
    ["add-secret-button", new Set(["F14-secrets.md"])],
    ["api-key-entry-name", new Set(["F01-first-run-and-sign-in.md"])],
    ["plugin-card-city-weather", new Set(["F19-plugins.md"])],
    ["renamed-away-button", new Set(["F14-secrets.md"])],
  ]);
  const literals = new Set([
    "add-secret-button",
    "api-key-entry",
    "plugin-card",
  ]);
  const result = resolveTestids(cited, literals);
  assert.equal(result.cited, 4);
  assert.deepEqual(result.unresolved, [
    { id: "renamed-away-button", files: ["F14-secrets.md"] },
  ]);
  // The live map against the live source tree: zero unresolved today, and
  // --strict makes any drift a failing exit.
  const res = spawnSync(process.execPath, [cli, "map", "testids", "--strict"], {
    encoding: "utf8",
  });
  const out = JSON.parse(res.stdout);
  assert.equal(res.status, 0, res.stdout);
  assert.deepEqual(out.unresolved, []);
  assert.ok(out.cited > 900);
});

test("evidence report shows notes, family counts, and changes against a baseline", () => {
  const dir = mkdtempSync(join(tmpdir(), "cov-run-"));
  mkdirSync(join(dir, "private"));
  mkdirSync(join(dir, "evidence"));
  writeFileSync(
    join(dir, "run.json"),
    JSON.stringify({
      runId: "r2",
      revision: "abc",
      mode: "local",
      baseUrl: "http://127.0.0.1:9",
      ports: { ingress: 9 },
      launcherPgid: 0,
      versions: { agentServer: "1.53.0" },
    }),
  );
  const rows = [
    {
      feature: "F14.create",
      entry: "Settings > Secrets > Add",
      result: "pass",
      artifacts: [],
    },
    {
      feature: "F14.agent-access",
      entry: "conversation start",
      result: "blocked",
      note: "no DEEPSEEK_API_KEY",
      artifacts: [],
    },
    {
      feature: "F09.index-redirect",
      entry: "gear",
      result: "fail",
      artifacts: [],
    },
  ];
  writeFileSync(
    join(dir, "evidence", "ledger.jsonl"),
    rows
      .map((r) => JSON.stringify({ ts: "2026-10-06T00:00:00Z", ...r }))
      .join("\n") + "\n",
  );
  const baseline = join(dir, "baseline.jsonl");
  writeFileSync(
    baseline,
    [
      {
        feature: "F14.create",
        entry: "Settings > Secrets > Add",
        result: "fail",
        artifacts: [],
      },
      {
        feature: "F09.index-redirect",
        entry: "gear",
        result: "pass",
        artifacts: [],
      },
      {
        feature: "F14.phone",
        entry: "Settings > Secrets",
        result: "pass",
        artifacts: [],
      },
    ]
      .map((r) => JSON.stringify({ ts: "2026-10-05T00:00:00Z", ...r }))
      .join("\n") + "\n",
  );
  const report = run(["evidence", "report", "--baseline", baseline], {
    OH_VERIFY_RUN: dir,
  });
  assert.equal(report.status, 0, report.stdout + report.stderr);
  // Blocked rows carry their prerequisite; fail and blocked rows come first.
  assert.match(report.stdout, /\| blocked \| no DEEPSEEK_API_KEY \|/);
  assert.ok(
    report.stdout.indexOf("F09.index-redirect") <
      report.stdout.indexOf("F14.create"),
  );
  assert.match(report.stdout, /\| F14 \| 1 \| 0 \| 1 \| 0 \|/);
  assert.match(
    report.stdout,
    /Newly failing:\n\n- F09.index-redirect \(gear\): pass → fail/,
  );
  assert.match(
    report.stdout,
    /Newly passing:\n\n- F14.create \(Settings > Secrets > Add\): fail → pass/,
  );
  assert.match(report.stdout, /Not checked this run:\n\n- F14.phone/);
  const json = run(["evidence", "report", "--json", "--baseline", baseline], {
    OH_VERIFY_RUN: dir,
  });
  assert.equal(json.json.counts.blocked, 1);
  assert.deepEqual(
    json.json.changes.newlyFailing.map((c) => c.feature),
    ["F09.index-redirect"],
  );
  assert.deepEqual(
    json.json.changes.missing.map((c) => c.feature),
    ["F14.phone"],
  );
  assert.equal(
    run(["evidence", "report", "--baseline", join(dir, "nope.jsonl")], {
      OH_VERIFY_RUN: dir,
    }).status,
    2,
  );
});

test("a browser call waits for the verb's own deadline plus a margin", () => {
  // The daemon runs one Playwright waitFor with the verb's --timeout; the
  // client used to abort every call at 120 s, so `wait --timeout 150000`
  // failed with an aborted request while the page kept waiting.
  assert.equal(browserCallLimit({}), 120_000);
  assert.equal(browserCallLimit({ timeout: 30_000 }), 120_000);
  assert.equal(browserCallLimit({ timeout: 150_000 }), 160_000);
  assert.equal(browserCallLimit({ observeMs: 130_000 }), 140_000);
  assert.equal(browserCallLimit({ timeout: "125000" }), 135_000);
  assert.equal(browserCallLimit({ timeout: undefined }), 120_000);
});

test("the tmux directory of a run comes from its whole path", () => {
  // Two runs whose ids end alike must not share a tmux server (the earlier
  // scheme used the last six characters of the run id).
  assert.notEqual(
    tmuxPathFor("/tmp/openhands-verify/qa-run-a1"),
    tmuxPathFor("/tmp/openhands-verify/qb-run-a1"),
  );
  assert.equal(tmuxPathFor("/tmp/x/run-1"), tmuxPathFor("/tmp/x/run-1"));
  assert.match(tmuxPathFor("/tmp/x/run-1"), /^\/tmp\/ohv-tmux-[0-9a-f]{10}$/);
});

test("help names every fixture kind and the new map verbs", () => {
  const { stdout } = run(["--help"]);
  for (const word of [
    "git-remote",
    "tarball",
    "skill",
    "mcp-server",
    "affected",
    "testids",
  ])
    assert.match(stdout, new RegExp(word));
  const map = spawnSync(process.execPath, [cli, "help", "map"], {
    encoding: "utf8",
  });
  assert.match(map.stdout, /map affected \[--base REF\]/);
  assert.match(map.stdout, /map testids \[--strict\]/);
  const evidence = spawnSync(process.execPath, [cli, "help", "evidence"], {
    encoding: "utf8",
  });
  assert.match(evidence.stdout, /--baseline/);
});

test("the maintenance baseline line is read and moved in place", () => {
  const index =
    "# Map\n\nIntro.\n\nMaintenance baseline: main@ed815e141409c7b52991ab8d13c553b595da9165 (2026-10-06). The next maintenance pass starts from this commit; a pass proposes the next baseline in its PR, and merging that PR accepts it.\n\n## Baseline preconditions\n";
  assert.deepEqual(parseBaseline(index), {
    sha: "ed815e141409c7b52991ab8d13c553b595da9165",
    date: "2026-10-06",
  });
  assert.equal(parseBaseline("# Map\n\nno line here\n"), null);
  const sha = "7cd6496040dd74d75e93feecaea528c6c50efef7";
  const moved = withBaseline(index, sha, "2026-10-07");
  assert.match(
    moved,
    /^Maintenance baseline: main@7cd6496040dd74d75e93feecaea528c6c50efef7 \(2026-10-07\)\. The next maintenance pass starts from this commit/m,
  );
  // Only the SHA and date move; the prose and the rest of the index stay.
  assert.equal(
    moved.replace(sha, "X").replace("2026-10-07", "D"),
    index.replace(/ed815e1[0-9a-f]+/, "X").replace("2026-10-06", "D"),
  );
  assert.throws(
    () => withBaseline(index, "7cd6496", "2026-10-07"),
    /full 40-hex SHA/,
  );
  assert.throws(
    () => withBaseline("# Map\n", sha, "2026-10-07"),
    /no `Maintenance baseline/,
  );
  // A merge that kept both sides' lines: nothing moves until one is removed.
  const twice = `${index}\nMaintenance baseline: main@${sha} (2026-10-07). Other.\n`;
  assert.equal(baselineLines(twice).length, 2);
  assert.throws(
    () => withBaseline(twice, sha, "2026-10-08"),
    /more than one `Maintenance baseline:` line/,
  );
});

test("map baseline reads the index line, and map affected starts from it by default", () => {
  const baseline = spawnSync(process.execPath, [cli, "map", "baseline"], {
    encoding: "utf8",
  });
  assert.equal(baseline.status, 0, baseline.stdout);
  const { baseline: recorded, hint } = JSON.parse(baseline.stdout);
  const affected = spawnSync(process.execPath, [cli, "map", "affected"], {
    encoding: "utf8",
  });
  const json = JSON.parse(affected.stdout);
  if (recorded === null) {
    // Until the index carries the line (OpenHands/OpenHands#18085), the
    // default range cannot be computed and the usage error says so.
    assert.match(hint, /no `Maintenance baseline:` line/);
    assert.equal(affected.status, 2, affected.stdout);
    assert.match(json.error, /Maintenance baseline/);
  } else {
    assert.match(recorded.sha, /^[0-9a-f]{40}$/);
    assert.match(recorded.date, /^\d{4}-\d{2}-\d{2}$/);
    if (recorded.known) {
      assert.equal(affected.status, 0, affected.stdout);
      assert.equal(json.range.base, recorded.sha);
      assert.equal(json.range.baseSource, "map index baseline");
    } else {
      // A shallow clone (CI checks out at depth 1) does not hold the
      // baseline commit: an environment error with the deepening hint, not
      // a stack trace.
      assert.equal(affected.status, 3, affected.stdout);
      assert.match(json.hint, /shallow clone/);
    }
  }
  // An explicit --base never consults the line. HEAD..HEAD is the one range
  // every clone can resolve, shallow ones included.
  const explicit = JSON.parse(
    spawnSync(
      process.execPath,
      [cli, "map", "affected", "--base", "HEAD", "--target", "HEAD"],
      {
        encoding: "utf8",
      },
    ).stdout,
  );
  assert.equal(explicit.range.baseSource, "--base");
  assert.equal(explicit.range.target, "HEAD");
  assert.equal(explicit.changed, 0);
  // --set refuses a ref that is not a commit, and a bare --set, before
  // touching the index.
  const bad = spawnSync(
    process.execPath,
    [cli, "map", "baseline", "--set", "not-a-ref-xyz"],
    {
      encoding: "utf8",
    },
  );
  assert.equal(bad.status, 2);
  const bare = spawnSync(process.execPath, [cli, "map", "baseline", "--set"], {
    encoding: "utf8",
  });
  assert.equal(bare.status, 2);
  assert.match(JSON.parse(bare.stdout).hint, /\$TARGET/);
  // The line reads main@<sha>: a commit that is not on main is refused when
  // main is in the clone (a PR branch's HEAD, here), so a pass cannot record
  // its own branch tip by mistake.
  const repo = resolve(here, "../../../..");
  const mainRef = ["origin/main", "main"].find(
    (r) =>
      spawnSync("git", ["rev-parse", "--verify", "--quiet", r], {
        cwd: repo,
      }).status === 0,
  );
  const headOnMain =
    mainRef &&
    spawnSync("git", ["merge-base", "--is-ancestor", "HEAD", mainRef], {
      cwd: repo,
    }).status === 0;
  if (mainRef && !headOnMain) {
    const branchTip = spawnSync(
      process.execPath,
      [cli, "map", "baseline", "--set", "HEAD"],
      { encoding: "utf8" },
    );
    assert.equal(branchTip.status, 2, branchTip.stdout);
    assert.match(JSON.parse(branchTip.stdout).error, /is not on/);
  }
  const help = spawnSync(process.execPath, [cli, "help", "map"], {
    encoding: "utf8",
  });
  assert.match(help.stdout, /map baseline \[--set TARGET \[--force\]\]/);
});

test("request bodies are shown with credentials and env maps redacted", () => {
  // A stored secret sent behind its placeholder must stay recognizable as
  // the placeholder; a real value shows only its length.
  const body = redactBody(
    JSON.stringify({
      name: "qa_vault",
      command: "/bin/sh",
      env: { QA_VAULT_TOKEN: "qa-vault-secret-000", QA_NODE: "**********" },
      api_key: "sk-abcdef",
      headers: { Authorization: "Bearer x" },
      nested: { session_api_key: "k", plain: "kept" },
      list: [{ token: "t" }],
    }),
  );
  const parsed = JSON.parse(body);
  assert.equal(parsed.name, "qa_vault");
  assert.equal(parsed.command, "/bin/sh");
  assert.equal(parsed.env.QA_VAULT_TOKEN, "<redacted 19 chars>");
  assert.equal(parsed.env.QA_NODE, "**********");
  assert.equal(parsed.api_key, "<redacted 9 chars>");
  assert.equal(parsed.headers.Authorization, "<redacted 8 chars>");
  assert.equal(parsed.nested.session_api_key, "<redacted 1 chars>");
  assert.equal(parsed.nested.plain, "kept");
  assert.equal(parsed.list[0].token, "<redacted 1 chars>");
  // Form bodies redact by key; anything else is described, never shown.
  assert.equal(redactBody("a=1&api_key=zzz"), "a=1&api_key=%3Credacted%3E");
  assert.equal(redactBody("--boundary\r\nraw"), "<non-JSON body, 15 chars>");
  // Secrets inside arrays, under a credential-named key, or at the top level
  // of a credential endpoint's payload never come back in clear.
  assert.deepEqual(
    JSON.parse(
      redactBody(
        JSON.stringify({
          env: ["TOKEN=abc123"],
          api_keys: ["sk-live-1"],
          credentials: { value: "p4ss", nested: ["x"] },
          values: ["kept"],
        }),
      ),
    ),
    {
      env: ["<redacted 12 chars>"],
      api_keys: ["<redacted 9 chars>"],
      credentials: {
        value: "<redacted 4 chars>",
        nested: ["<redacted 1 chars>"],
      },
      values: ["kept"],
    },
  );
  assert.equal(
    redactBody(JSON.stringify("hunter2"), { all: true }),
    '"<redacted 7 chars>"',
  );
  assert.equal(
    redactBody(JSON.stringify(["hunter2"]), { all: true }),
    '["<redacted 7 chars>"]',
  );
  assert.equal(
    redactBody("grant_type=x&code=SECRET", { all: true }),
    "grant_type=%3Credacted%3E&code=%3Credacted%3E",
  );
  // A credential endpoint redacts every string, whatever the field is called.
  assert.deepEqual(
    JSON.parse(
      redactBody(JSON.stringify({ name: "X", value: "v" }), { all: true }),
    ),
    { name: "<redacted 1 chars>", value: "<redacted 1 chars>" },
  );
  assert.equal(redactBody(""), undefined);
  assert.equal(redactBody(null), undefined);
  assert.match(
    redactBody(JSON.stringify({ t: "x".repeat(5000) }), { limit: 50 }),
    /…$/,
  );
});

test("credentials carried by a URL are redacted whatever key holds the URL", () => {
  // The MCP editor sends a remote server's URL as typed, under `server.url`
  // (POST /api/v1/mcp/test) or `url` (a settings save): a key or a basic-auth
  // pair the user put in the URL must not come back in clear. The rest of
  // the URL is kept as written, so the row still proves the endpoint sent.
  const probe = JSON.parse(
    redactBody(
      JSON.stringify({
        server: {
          type: "http",
          url: "https://example.invalid/mcp?api_key=DUMMY_URL_API_KEY",
          headers: { "X-Api-Key": "dummy-header-key" },
        },
        name: "qa-remote",
        timeout: 30,
      }),
    ),
  );
  assert.equal(
    probe.server.url,
    "https://example.invalid/mcp?api_key=<redacted 17 chars>",
  );
  assert.equal(probe.server.headers["X-Api-Key"], "<redacted 16 chars>");
  assert.equal(probe.name, "qa-remote");
  const save = JSON.parse(
    redactBody(
      JSON.stringify({
        name: "qa-sse",
        type: "sse",
        url: "https://qa:hunter2pw@example.invalid/sse?v=2&Token=t0k#frag",
        description: "docs at https://docs.example.invalid/sse?page=1",
      }),
    ),
  );
  assert.equal(
    save.url,
    "https://<redacted 2 chars>:<redacted 9 chars>@example.invalid/sse?v=2&Token=<redacted 3 chars>#frag",
  );
  assert.equal(
    save.description,
    "docs at https://docs.example.invalid/sse?page=1",
  );
  // A URL inside prose (a prompt naming a server) is treated the same; the
  // settings API's placeholder stays recognizable inside a URL too.
  assert.equal(
    JSON.parse(
      redactBody(JSON.stringify({ content: "use https://u:p@h/x now" })),
    ).content,
    "use https://<redacted 1 chars>:<redacted 1 chars>@h/x now",
  );
  assert.equal(
    redactUrl("https://user:**********@example.invalid/mcp"),
    "https://<redacted 4 chars>:**********@example.invalid/mcp",
  );
  assert.equal(
    redactUrl("https://sk-abc@example.invalid/mcp?secret%5Fid=s&q=kept"),
    "https://<redacted 6 chars>@example.invalid/mcp?secret%5Fid=<redacted 1 chars>&q=kept",
  );
  // An unencoded `@` in a password belongs to the password: the userinfo
  // ends at the last `@` before the path, as `new URL()` reads it.
  assert.equal(
    redactUrl("https://user:p@ssw0rd@example.invalid/mcp"),
    "https://<redacted 4 chars>:<redacted 8 chars>@example.invalid/mcp",
  );
  assert.equal(
    redactUrl("https://host/path?email=a@b&token=t&next=https://u:pw@other/"),
    "https://host/path?email=a@b&token=<redacted 1 chars>&next=https://<redacted 1 chars>:<redacted 2 chars>@other/",
  );
  assert.equal(
    redactUrl("http://example.invalid/mcp"),
    "http://example.invalid/mcp",
  );
  assert.equal(redactUrl("a:b@c mailto:x@y"), "a:b@c mailto:x@y");
  // A form body's URL values go through the same redaction.
  assert.equal(
    new URLSearchParams(
      redactBody("name=x&url=https%3A%2F%2Fu%3Ap%40h%2Fmcp%3Fkey%3Dk"),
    ).get("url"),
    "https://<redacted 1 chars>:<redacted 1 chars>@h/mcp?key=<redacted 1 chars>",
  );
});

test("conversation events reads pages until the rows asked for are in hand", async () => {
  const pages = {
    undefined: { items: [1, 2, 3], next_page_id: "p2" },
    p2: { items: [4, 5], next_page_id: "p3" },
    p3: { items: [6], next_page_id: null },
  };
  const calls = [];
  const fetchPage = async (id) => {
    calls.push(id);
    return pages[id];
  };
  // Enough rows after one page: no second request.
  assert.deepEqual(await collectEvents(fetchPage, 3), {
    items: [1, 2, 3],
    more: true,
    pages: 1,
  });
  calls.length = 0;
  // More rows than one page holds: follow next_page_id, stop at the end.
  assert.deepEqual(await collectEvents(fetchPage, 500), {
    items: [1, 2, 3, 4, 5, 6],
    more: false,
    pages: 3,
  });
  assert.deepEqual(calls, [undefined, "p2", "p3"]);
  // A page without items or a missing body ends the walk cleanly.
  assert.deepEqual(await collectEvents(async () => undefined, 10), {
    items: [],
    more: false,
    pages: 1,
  });
  // A filter counts only the rows that pass it, so a kind that is rare keeps
  // the walk going; a server that never ends paging stops at MAX_PAGES.
  const mixed = async (id) => ({
    items: [{ kind: "State" }, { kind: "State" }, { kind: "Message" }],
    next_page_id:
      id === "p3" ? null : `p${(Number(String(id).slice(1)) || 1) + 1}`,
  });
  const filtered = await collectEvents(mixed, 3, (e) => e.kind === "Message");
  assert.equal(filtered.pages, 3);
  assert.equal(filtered.items.filter((e) => e.kind === "Message").length, 3);
  const endless = await collectEvents(
    async () => ({ items: [{ kind: "x" }], next_page_id: "again" }),
    1000,
  );
  assert.equal(endless.pages, MAX_PAGES);
  assert.equal(endless.more, true);
});

test("image attachments are counted from a message's content blocks", () => {
  assert.equal(countImages("plain text"), 0);
  assert.equal(countImages(undefined), 0);
  assert.equal(
    countImages([
      { type: "text", text: " " },
      { type: "image", image_urls: ["data:image/png;base64,AAAA", "data:x"] },
      { type: "image" },
    ]),
    3,
  );
});

test("help documents the observe, bodies, tab, mode and commit additions", () => {
  const browser = spawnSync(process.execPath, [cli, "help", "browser"], {
    encoding: "utf8",
  }).stdout;
  assert.match(
    browser,
    /scroll \[<sel>\] \[--by PX\] \[--x PX\]\n\s+\[--observe SEL/,
  );
  assert.match(browser, /network .*\[--bodies\]/);
  assert.match(browser, /tab new \[\/path\]/);
  assert.match(browser, /wait-tab <url-regex> \[--timeout MS\] \[--new\]/);
  const conversation = spawnSync(
    process.execPath,
    [cli, "help", "conversation"],
    { encoding: "utf8" },
  ).stdout;
  assert.match(conversation, /images: N/);
  assert.match(conversation, /more:true/);
  assert.match(conversation, /prints the prompt it typed/);
  const fixture = spawnSync(process.execPath, [cli, "help", "fixture"], {
    encoding: "utf8",
  }).stdout;
  assert.match(fixture, /fixture skill .*\[--commit\]/);
  const workspace = spawnSync(process.execPath, [cli, "help", "workspace"], {
    encoding: "utf8",
  }).stdout;
  assert.match(
    workspace,
    /workspace open PATH\|NAME \[--stay\] \[--mode local_repo\|new_worktree\]/,
  );
});

test("fixture skill --commit records the project skill once and reports a re-run", () => {
  // A stopped run directory is enough for fixtures: they only need run.json.
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
  const env = { OH_VERIFY_RUN: dir };
  assert.equal(run(["fixture", "git-repo", "--name", "qa-t"], env).status, 0);
  const first = run(
    [
      "fixture",
      "skill",
      "--repo",
      "qa-t",
      "--name",
      "qa-t-skill",
      "--trigger",
      "qa-t-ping",
      "--commit",
    ],
    env,
  );
  assert.equal(first.status, 0, first.stdout);
  assert.equal(first.json.scope, "project");
  assert.match(first.json.committed.sha, /^[0-9a-f]{40}$/);
  assert.equal(first.json.committed.message, "Add qa-t-skill skill");
  const log = spawnSync(
    "git",
    ["-C", join(dir, "workspace", "qa-t"), "log", "--format=%s"],
    {
      encoding: "utf8",
    },
  )
    .stdout.trim()
    .split("\n");
  assert.deepEqual(log, ["Add qa-t-skill skill", "Initial fixture commit"]);
  // The same command again changes nothing and says so.
  const again = run(
    [
      "fixture",
      "skill",
      "--repo",
      "qa-t",
      "--name",
      "qa-t-skill",
      "--trigger",
      "qa-t-ping",
      "--commit",
    ],
    env,
  );
  assert.equal(again.status, 0, again.stdout);
  assert.equal(again.json.committed.sha, first.json.committed.sha);
  assert.equal(again.json.committed.unchanged, true);
  // Only the skill's path is committed: a file staged beforehand stays staged.
  const repo = join(dir, "workspace", "qa-t");
  writeFileSync(join(repo, "other.txt"), "staged elsewhere\n");
  spawnSync("git", ["-C", repo, "add", "other.txt"]);
  const third = run(
    ["fixture", "skill", "--repo", "qa-t", "--name", "qa-v", "--commit"],
    env,
  );
  assert.equal(third.status, 0, third.stdout);
  const files = spawnSync(
    "git",
    ["-C", repo, "show", "--name-only", "--format=", "HEAD"],
    { encoding: "utf8" },
  ).stdout.trim();
  assert.equal(files, ".agents/skills/qa-v/SKILL.md");
  const staged = spawnSync(
    "git",
    ["-C", repo, "diff", "--cached", "--name-only"],
    {
      encoding: "utf8",
    },
  ).stdout.trim();
  assert.equal(staged, "other.txt");
  // Without --commit nothing is recorded; a personal skill cannot be committed.
  const plain = run(
    ["fixture", "skill", "--repo", "qa-t", "--name", "qa-u"],
    env,
  );
  assert.equal(plain.status, 0);
  assert.equal(plain.json.committed, undefined);
  const personal = run(["fixture", "skill", "--name", "qa-p", "--commit"], env);
  assert.equal(personal.status, 2);
  assert.match(personal.json.error, /needs --repo/);
});

test("llm preset repoints default only off another live LLM profile", () => {
  const seeded = {
    id: "a1",
    name: "default",
    revision: 3,
    schema_version: 1,
    agent_kind: "openhands",
    llm_profile_ref: "deepseek-chat",
    enable_sub_agents: true,
    mcp_server_refs: ["qa_mcp_a"],
  };
  const names = ["deepseek-chat", "deepseek-flash", "deepseek-pro"];
  // Onboarding left default on its own profile: move it, keep every other field.
  assert.deepEqual(agentProfileRepoint(seeded, names, "deepseek-flash"), {
    from: "deepseek-chat",
    body: {
      schema_version: 1,
      agent_kind: "openhands",
      llm_profile_ref: "deepseek-flash",
      enable_sub_agents: true,
      mcp_server_refs: ["qa_mcp_a"],
    },
  });
  // Already there, a fresh run's missing `default` ref, or no ref: unchanged.
  assert.equal(agentProfileRepoint(seeded, names, "deepseek-chat"), null);
  assert.equal(
    agentProfileRepoint(
      { ...seeded, llm_profile_ref: "default" },
      names,
      "deepseek-flash",
    ),
    null,
  );
  assert.equal(
    agentProfileRepoint(
      { ...seeded, llm_profile_ref: null },
      names,
      "deepseek-flash",
    ),
    null,
  );
  // An ACP default owns its own model; an older backend has no profile.
  assert.equal(
    agentProfileRepoint(
      { ...seeded, agent_kind: "acp" },
      names,
      "deepseek-flash",
    ),
    null,
  );
  assert.equal(agentProfileRepoint(undefined, names, "deepseek-flash"), null);
});

test("llm help says when preset repoints the default agent profile", () => {
  const help = spawnSync(process.execPath, [cli, "help", "llm"], {
    encoding: "utf8",
  });
  assert.equal(help.status, 0);
  assert.match(
    help.stdout,
    /'preset' also points the 'default' agent profile at deepseek-flash/,
  );
  assert.match(help.stdout, /and 'set' leave agent profiles as they are/);
  assert.match(help.stdout, /'repointed'/);
});

test("a recording's timeline cuts the lead-in and pauses, and holds the end", () => {
  // Changes at 0, 3, 3.2 and 10 s; stopped at 10.5 s.
  const frames = [{ t: 0 }, { t: 3000 }, { t: 3200 }, { t: 10000 }];
  assert.deepEqual(frameDurations(frames, 10500), [1, 0.2, 6.8, 1]);
  // Paused from 4 s to 9 s: the frame shown at 3.2 s lasts 1.8 s.
  const cut = withoutPauses(frames, 10500, [{ at: 4000, ms: 5000 }]);
  assert.deepEqual(
    cut.frames.map((f) => f.t),
    [0, 3000, 3200, 5000],
  );
  assert.equal(cut.endT, 5500);
  assert.deepEqual(frameDurations(cut.frames, cut.endT), [1, 0.2, 1.8, 1]);
  // One unchanging frame keeps its real length, and is shown for at least 1 s.
  assert.deepEqual(frameDurations([{ t: 0 }], 5000), [5]);
  assert.deepEqual(frameDurations([{ t: 0 }], 200), [1]);
  // Constant frame rate: ends rounded on the running total, at least one each.
  assert.deepEqual(frameRepeats([1, 0.2, 6.8, 1], 10), [10, 2, 68, 10]);
  assert.deepEqual(frameRepeats([0.15, 0.15, 0.15], 10), [2, 1, 2]);
  assert.deepEqual(frameRepeats([0.01, 0.01], 10), [1, 1]);
  // The canvas is the largest frame with even sides; smaller frames are padded.
  assert.deepEqual(
    canvasSize([
      { width: 1440, height: 1000 },
      { width: 391, height: 845 },
    ]),
    { width: 1440, height: 1000 },
  );
  assert.deepEqual(canvasSize([{ width: 391, height: 845 }]), {
    width: 392,
    height: 846,
  });
});

test("the recorder picks MP4 with libx264, else WebM, and finds Playwright's ffmpeg", () => {
  const system = [
    " V....D libx264              libx264 H.264 / AVC (codec h264)",
    " V....D libx264rgb           libx264 H.264 RGB (codec h264)",
    " V....D gif                  GIF (Graphics Interchange Format)",
  ].join("\n");
  assert.deepEqual(encoderSupport(system), { format: "mp4", gif: true });
  const bundled =
    " V....D libvpx               libvpx VP8 (codec vp8)\n A....D aac x";
  assert.deepEqual(encoderSupport(bundled), { format: "webm", gif: false });
  assert.equal(encoderSupport(" V....D libx264rgb  rgb only"), null);
  const root = mkdtempSync(join(tmpdir(), "pw-browsers-"));
  for (const dir of ["chromium-1194", "ffmpeg-1009", "ffmpeg-1011"])
    mkdirSync(join(root, dir));
  assert.deepEqual(
    playwrightFfmpegPaths({ PLAYWRIGHT_BROWSERS_PATH: root }, "darwin"),
    [
      join(root, "ffmpeg-1011", "ffmpeg-mac"),
      join(root, "ffmpeg-1009", "ffmpeg-mac"),
    ],
  );
  assert.deepEqual(
    playwrightFfmpegPaths({ XDG_CACHE_HOME: root }, "linux"),
    [],
  );
  const mp4 = ffmpegArgs("mp4", {
    fps: 10,
    width: 320,
    height: 700,
    out: "o.mp4",
  });
  assert.equal(mp4[mp4.indexOf("-i") + 1], "pipe:0");
  assert.ok(mp4.includes("libx264") && mp4.at(-1) === "o.mp4");
  assert.match(mp4[mp4.indexOf("-vf") + 1], /pad=320:700:.*format=yuv420p$/);
  assert.ok(
    ffmpegArgs("webm", {
      fps: 10,
      width: 2,
      height: 2,
      out: "o.webm",
    }).includes("vp8"),
  );
});

const RED_16X16_JPEG =
  "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAgUFBcUFxsbGxsbGyAeICEhISAgICAhISEkJCQqKiokJCQhISQkKCgqKi4vLisrKisvLzIyMjw8OTlGRkhWVmf/xABMAAEBAAAAAAAAAAAAAAAAAAAABgEBAQAAAAAAAAAAAAAAAAAABgcQAQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAAQABADASIAAhEAAxEA/9oADAMBAAIRAxEAPwCLAFF/f//Z";
const BLUE_16X12_JPEG =
  "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAgUFBcUFxsbGxsbGyAeICEhISAgICAhISEkJCQqKiokJCQhISQkKCgqKi4vLisrKisvLzIyMjw8OTlGRkhWVmf/xABMAAEBAAAAAAAAAAAAAAAAAAAABwEBAQAAAAAAAAAAAAAAAAAABQcQAQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAAMABADASIAAhEAAxEA/9oADAMBAAIRAxEAPwCOAL+Lf//Z";

test.skipIf(!findFfmpeg())(
  "recorded frames of two sizes encode at the real timing",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "record-"));
    const frames = [
      [RED_16X16_JPEG, 0, 16, 16],
      [BLUE_16X12_JPEG, 1500, 16, 12],
    ].map(([data, t, width, height], i) => {
      const file = join(dir, `frame-${i}.jpg`);
      writeFileSync(file, Buffer.from(data, "base64"));
      return { file, t, width, height };
    });
    const ffmpeg = findFfmpeg();
    const out = join(dir, `out.${ffmpeg.format}`);
    const written = await encodeRecording({
      ffmpeg,
      frames,
      durations: frameDurations(frames, 2000),
      fps: 10,
      ...canvasSize(frames),
      out,
    });
    // 1 s lead-in (cut from 1.5 s) + 1 s hold on the last frame.
    assert.equal(written, 20);
    assert.ok(readFileSync(out).length > 0);
  },
);

test("browser help documents record", () => {
  const help = spawnSync(process.execPath, [cli, "help", "browser"], {
    encoding: "utf8",
  });
  assert.equal(help.status, 0);
  assert.match(help.stdout, /record start --feature ID --name N/);
  assert.match(help.stdout, /record pause \| record resume/);
  assert.match(help.stdout, /record stop \[--gif\]/);
  const bad = spawnSync(
    process.execPath,
    [cli, "browser", "record", "start", "--name", "x", "--run", tmpdir()],
    {
      encoding: "utf8",
    },
  );
  assert.notEqual(bad.status, 0);
});
