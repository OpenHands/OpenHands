const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

// Execute the actual github-script body; only the GitHub REST transport is faked.
const workflow = fs.readFileSync(
  path.join(__dirname, "../../workflows/pr-artifacts.yml"), "utf8",
);
const match = workflow.match(/^([ \t]*)script: \|\n([\s\S]*)$/m);
assert.ok(match, "workflow must contain an inline github-script body");
const script = match[2].split("\n")
  .map((line) => line.slice(match[1].length + 2)).join("\n");
const run = new (Object.getPrototypeOf(async function () {}).constructor)(
  "github", "context", "core", script,
);
const file = (filename, status = "modified", extra = {}) => ({ filename, status, ...extra });
const marker = "<!-- pr-artifacts-notice -->";

async function notice(files, comments = []) {
  const writes = [];
  const listFiles = Symbol("listFiles");
  const listComments = Symbol("listComments");
  const github = {
    rest: {
      pulls: { listFiles },
      issues: {
        listComments,
        createComment: async ({ issue_number, body }) => writes.push({ issue_number, body }),
        updateComment: async ({ comment_id, body }) => writes.push({ comment_id, body }),
      },
    },
    paginate: async (endpoint, params) => {
      assert.equal(params.per_page, 100);
      if (endpoint === listFiles) {
        assert.equal(params.pull_number, 42);
        return files;
      }
      assert.equal(endpoint, listComments);
      assert.equal(params.issue_number, 42);
      return comments;
    },
  };
  await run(github, { repo: { owner: "org", repo: "repo" }, issue: { number: 42 } }, {
    warning: () => {},
  });
  return writes;
}

for (const [name, artifact] of [
  ["added", file(".pr/design.html", "added")],
  ["modified", file(".pr/live-e2e/1/live-agent-response.png")],
  ["renamed-in", file(".pr/notes.md", "renamed", { previous_filename: "docs/notes.md" })],
]) {
  test(`${name} .pr/ file posts the notice`, async () => {
    const [comment, ...rest] = await notice([file("src/app.ts"), artifact]);
    assert.deepEqual(rest, []);
    assert.equal(comment.issue_number, 42);
    assert.match(comment.body, /next release PR removes them/);
  });
}

test("removed, renamed-out, and lookalike paths post nothing", async () => {
  assert.deepEqual(await notice([
    file(".pr/old.png", "removed"),
    file("docs/notes.md", "renamed", { previous_filename: ".pr/notes.md" }),
    file(".pr-other/notes.md", "added"),
    file("src/.pr/notes.md", "added"),
    file("package.json"),
  ]), []);
});

test("an existing notice is updated in place", async () => {
  const comments = [{ id: 7, body: `${marker}\nold` }];
  const [kept] = await notice([file(".pr/notes.md")], comments);
  const [cleared] = await notice([file(".pr/notes.md", "removed")], comments);
  assert.equal(kept.comment_id, 7);
  assert.match(kept.body, /PR Artifacts Notice/);
  assert.equal(cleared.comment_id, 7);
  assert.match(cleared.body, /PR Artifacts Cleaned Up/);
});
