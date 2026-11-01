const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

// Execute the actual github-script body, without checking out or running PR code.
const workflow = fs.readFileSync(
  path.join(__dirname, "../../workflows/verify-skill-isolation.yml"), "utf8",
);
const match = workflow.match(/^([ \t]*)script: \|\n([\s\S]*)$/m);
assert.ok(match, "workflow must contain an inline github-script body");
const script = match[2].split("\n")
  .map((line) => line.slice(match[1].length + 4)).join("\n");
const run = new (Object.getPrototypeOf(async function () {}).constructor)(
  "github", "context", "core", script,
);
const verify = ".agents/skills/verify-openhands/SKILL.md";
const file = (filename, extra = {}) => ({ filename, status: "modified", ...extra });

async function check(files, options = {}) {
  const errors = [];
  let reads = 0;
  const pr = {
    head: { sha: "head" }, base: { sha: "base" },
    changed_files: options.count ?? files.length,
  };
  const get = async () => ({ data: ++reads === 2 && options.after ? options.after : pr });
  const listFiles = Symbol("listFiles");
  const github = {
    rest: { pulls: { get, listFiles } },
    paginate: async (endpoint, params) => {
      assert.equal(endpoint, listFiles);
      assert.equal(params.per_page, 100);
      assert.equal(params.pull_number, 42);
      if (options.apiError) throw new Error("GitHub unavailable");
      return files;
    },
  };
  await run(github, { repo: { owner: "org", repo: "repo" }, issue: { number: 42 } }, {
    info: () => {}, setFailed: (message) => errors.push(message),
  });
  return errors;
}

for (const [name, files, allowed] of [
  ["product-only", [file("src/app.py"), file("README.md")], true],
  ["AGENTS-only", [file("AGENTS.md"), file("nested/AGENTS.md")], true],
  ["skill-only additions and deletions", [file(verify, { status: "added" }),
    file(".agents/skills/verify-sdk/map.md", { status: "removed" })], true],
  ["skill plus AGENTS", [file(verify), file("nested/AGENTS.md")], true],
  ["mixed product and skill", [file(verify), file("src/app.py")], false],
  ["mixed other skill", [file(verify), file(".agents/skills/review/SKILL.md")], false],
  ["skill plus .pr evidence", [file(verify), file(".pr/map-maintenance/report.md")], true],
  ["nested .pr is not exempt", [file(verify), file("src/.pr/notes.md")], false],
  ["product plus .pr evidence", [file("src/app.py"), file(".pr/18160/proof.json")], true],
  ["README is not exempt", [file(verify), file("README.md")], false],
  ["lowercase agents is not exempt", [file(verify), file("agents.md")], false],
  ["verify prefix without dash is ordinary product", [file(".agents/skills/verify/SKILL.md"), file("src/app.py")], true],
  ["AGENTS suffix is not exempt", [file(verify), file("nested/NOT_AGENTS.md")], false],
  ["nested lookalike is ordinary product", [file("docs/.agents/skills/verify-x/a")], true],
  ["rename out of skill", [file("docs/map.md", { status: "renamed", previous_filename: verify })], false],
  ["rename into skill", [file(verify, { status: "renamed", previous_filename: "src/app.py" })], false],
  ["rename inside skills", [file(verify, { status: "renamed", previous_filename:
    ".agents/skills/verify-automation/SKILL.md" })], true],
  ["AGENTS renamed to product with skill", [file(verify), file("src/app.py", {
    status: "renamed", previous_filename: "AGENTS.md" })], false],
  ["AGENTS rename inside skill", [file(verify, { status: "renamed", previous_filename: "AGENTS.md" })], true],
  ["rename metadata missing", [file(verify, { status: "renamed" })], false],
]) {
  test(name, async () => assert.equal((await check(files)).length === 0, allowed));
}

test("more than one page still detects a mixed PR", async () => {
  const files = Array.from({ length: 101 }, (_, i) => file(`src/${i}.py`));
  files.push(file(verify));
  assert.equal((await check(files)).length, 1);
});
test("incomplete file listing fails closed", async () => {
  assert.match((await check([file(verify)], { count: 3001 }))[0], /complete/i);
});
test("API failures fail closed", async () => {
  assert.match((await check([], { apiError: true }))[0], /unavailable/);
});
test("PR changes during pagination fail closed", async () => {
  assert.match((await check([file(verify)], { after: {
    head: { sha: "new-head" }, base: { sha: "base" }, changed_files: 1,
  } }))[0], /changed/i);
});

test("base changes during pagination fail closed", async () => {
  assert.match((await check([file(verify)], { after: {
    head: { sha: "head" }, base: { sha: "new-base" }, changed_files: 1,
  } }))[0], /changed/i);
});
test("file count changes during pagination fail closed", async () => {
  assert.match((await check([file(verify)], { after: {
    head: { sha: "head" }, base: { sha: "base" }, changed_files: 2,
  } }))[0], /changed/i);
});

test("a mixed PR's failure says how to split it and lists both sides", async () => {
  const [message] = await check([file(verify), file("src/app.py")]);
  assert.match(message, /split this PR/);
  assert.match(message, new RegExp(`- ${verify.replaceAll(".", "\\.")}`));
  assert.match(message, /- src\/app\.py/);
});
