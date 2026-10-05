// Fast checks for control-openhands that need no browser or running stack.
// They run with the rest of the suite (`npm test`), or alone:
//   npx vitest run .agents/skills/verify-openhands
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "vitest";
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
