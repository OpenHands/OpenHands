// Identity of a checkout's frontend build inputs: the committed tree, the
// uncommitted changes, and untracked (but not ignored) files under them. A
// new source file is part of the build as soon as it exists, so leaving
// untracked files out would reuse a stale build and report it as current.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { BUILD_INPUTS } from "./build-inputs.mjs";

export function buildIdentity(root, inputs = BUILD_INPUTS) {
  const git = (args) =>
    execFileSync("git", args, { cwd: root, encoding: "utf8" });
  const tree = git(["ls-tree", "HEAD", "--", ...inputs]);
  const dirty = git(["diff", "HEAD", "--", ...inputs]);
  const untracked = git([
    "ls-files",
    "--others",
    "--exclude-standard",
    "-z",
    "--",
    ...inputs,
  ])
    .split("\0")
    .filter(Boolean)
    .sort();
  const hash = createHash("sha256").update(tree).update(dirty);
  for (const file of untracked) {
    hash.update(`\0${file}\0`).update(readFileSync(join(root, file)));
  }
  const id = hash.digest("hex").slice(0, 16);
  return dirty.trim() || untracked.length ? `${id}-dirty` : id;
}
