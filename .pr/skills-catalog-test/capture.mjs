// Evidence for the same-name skills test fix. Runs the skills settings tests
// twice: with the test file from origin/main (before) and from this branch
// (after). Saves each run's raw output and renders it to a PNG; the only edits
// are stripped ANSI colors, dropped Vite "Sourcemap ... points to missing
// source files" warnings, and the repo path shown as ".". DEBUG_PRINT_LIMIT=0
// keeps Testing Library from appending the whole rendered DOM to the failure.
// Usage, from the repo root on a clean tree: node .pr/skills-catalog-test/capture.mjs
import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const root = process.cwd();
const outDir = path.join(root, ".pr/skills-catalog-test");
const testFile = "__tests__/routes/skills-settings.test.tsx";
const testPath = path.join(root, testFile);
const vitestArgs = ["vitest", "run", testFile];

const git = (...args) => execFileSync("git", args, { encoding: "utf8" });
const mainSha = git("rev-parse", "--short", "origin/main").trim();
const headSha = git("rev-parse", "--short", "HEAD").trim();

const runTests = () => {
  const run = spawnSync("npx", vitestArgs, {
    encoding: "utf8",
    env: {
      ...process.env,
      NO_COLOR: "1",
      FORCE_COLOR: "0",
      DEBUG_PRINT_LIMIT: "0",
    },
  });
  // eslint-disable-next-line no-control-regex
  const output = `${run.stdout}${run.stderr}`.replace(/\x1b\[[0-9;]*m/g, "");
  return output
    .replaceAll(root, ".")
    .split("\n")
    .filter(
      (line) =>
        !/^Sourcemap for ".*" points to missing source files$/.test(line),
    )
    .join("\n");
};

const branchTest = readFileSync(testPath, "utf8");
let before;
try {
  writeFileSync(testPath, git("show", `origin/main:${testFile}`));
  before = runTests();
} finally {
  writeFileSync(testPath, branchTest);
}
const after = runTests();

const runs = [
  {
    name: "before",
    caption: `before: ${testFile} from origin/main ${mainSha}`,
    output: before,
  },
  { name: "after", caption: `after: branch head ${headSha}`, output: after },
];

const require = createRequire(path.join(root, "package.json"));
const { chromium } = require("playwright");
const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;");

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2 });
for (const run of runs) {
  const text = `# ${run.caption}\n$ DEBUG_PRINT_LIMIT=0 npx ${vitestArgs.join(" ")}\n${run.output}`;
  writeFileSync(path.join(outDir, `${run.name}.log`), text);
  await page.setContent(
    `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#1e1e1e">` +
      `<pre style="margin:0;padding:16px 20px;width:1040px;color:#d4d4d4;` +
      `font:13px/1.45 Menlo,Consolas,monospace;white-space:pre-wrap">${escape(text)}</pre>`,
  );
  await page
    .locator("pre")
    .screenshot({ path: path.join(outDir, `${run.name}.png`) });
}
await browser.close();
