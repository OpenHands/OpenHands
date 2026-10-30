import { execSync } from "node:child_process";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  symlinkSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  buildWorkspaceFileListCommand,
  DEFAULT_FILE_DISCOVERY,
  parseWorkspaceFileList,
} from "#/utils/workspace-file-discovery";

// @spec WFD-001 — Configurable local workspace discovery
describe("workspace file discovery", () => {
  let workspace: string;
  beforeEach(() => {
    workspace = mkdtempSync(join(tmpdir(), "workspace-discovery-"));
    for (const directory of [
      "bin",
      "nested/bin",
      "obj",
      "node_modules",
      "quoted'$(touch injected)",
    ]) {
      mkdirSync(join(workspace, directory), { recursive: true });
      writeFileSync(join(workspace, directory, "file.txt"), "fixture");
    }
    writeFileSync(join(workspace, "instructions.md"), "fixture");
    symlinkSync("instructions.md", join(workspace, "AGENTS.md"));
    symlinkSync("nested", join(workspace, "directory-link"));
  });
  afterEach(() => rmSync(workspace, { recursive: true, force: true }));

  const list = (options = DEFAULT_FILE_DISCOVERY) =>
    parseWorkspaceFileList(
      execSync(buildWorkspaceFileListCommand(options), {
        cwd: workspace,
        encoding: "utf8",
      }),
      options.maxFiles,
    );

  it("preserves default exclusions and omits symlinks", () => {
    const result = list();
    expect(result.paths).toContain("bin/file.txt");
    expect(result.paths).not.toContain("node_modules/file.txt");
    expect(result.paths).not.toContain("AGENTS.md");
    expect(result.isTruncated).toBe(false);
  });

  it("prunes configured directories safely while preserving root bin and including file symlinks", () => {
    const result = list({
      maxFiles: 0,
      includeSymlinks: true,
      excludedPatterns: ["*/bin", "obj", "quoted'$(touch injected)"],
    });
    expect(result.paths).toEqual([
      "AGENTS.md",
      "bin/file.txt",
      "instructions.md",
      "node_modules/file.txt",
    ]);
  });

  it("detects overflow without warning at an exact limit and removes the cap when unlimited", () => {
    for (let index = 0; index < 2001; index += 1) {
      writeFileSync(join(workspace, `${index}.txt`), "fixture");
    }
    expect(list().paths).toHaveLength(2000);
    expect(list().isTruncated).toBe(true);
    const unlimited = list({ ...DEFAULT_FILE_DISCOVERY, maxFiles: 0 });
    expect(unlimited.paths.length).toBeGreaterThan(2000);
    expect(unlimited.isTruncated).toBe(false);
    const exact = list({
      ...DEFAULT_FILE_DISCOVERY,
      maxFiles: unlimited.paths.length,
    });
    expect(exact).toEqual(unlimited);
  });
});
