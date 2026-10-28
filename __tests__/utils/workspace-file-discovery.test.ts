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
  buildWindowsWorkspaceFileListCommand,
  buildWorkspaceFileListCommand,
  DEFAULT_FILE_DISCOVERY,
  parseWindowsWorkspaceFileList,
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

describe("Windows workspace discovery", () => {
  const options = {
    ...DEFAULT_FILE_DISCOVERY,
    excludedPatterns: ["node_modules", ".git", "src/generated"],
    maxFiles: 3,
  };

  it("builds a cmd.exe command with no POSIX-only tools", () => {
    const command = buildWindowsWorkspaceFileListCommand(
      options,
      "C:\\Users\\me\\proj",
    );
    expect(command).toBe(
      'dir /b /s /a:-d 2>nul | findstr /v /i /c:"\\node_modules\\" /c:"\\.git\\" /c:"\\src\\generated\\" | sort',
    );
    expect(command).not.toMatch(/head|\/dev\/null|^find /);
  });

  it("skips an exclusion that already appears in the workspace root", () => {
    const command = buildWindowsWorkspaceFileListCommand(
      options,
      "C:\\build\\node_modules\\proj",
    );
    expect(command).not.toContain("node_modules");
    expect(command).toContain('/c:"\\.git\\"');
  });

  it("drops patterns that cmd.exe could misread", () => {
    const command = buildWindowsWorkspaceFileListCommand(
      { ...options, excludedPatterns: ['a"b', "50%", "ok"] },
      "C:\\proj",
    );
    expect(command).toBe(
      'dir /b /s /a:-d 2>nul | findstr /v /i /c:"\\ok\\" | sort',
    );
  });

  it("returns forward-slash paths relative to the working dir", () => {
    const stdout = [
      "C:\\proj\\README.md",
      "C:\\proj\\src\\index.ts",
      "C:\\proj\\node_modules\\x\\a.js",
      "",
    ].join("\r\n");
    expect(parseWindowsWorkspaceFileList(stdout, "C:/proj/", options)).toEqual({
      paths: ["README.md", "src/index.ts"],
      isTruncated: false,
    });
  });

  it("caps the list and reports truncation", () => {
    const stdout = ["a", "b", "c", "d"]
      .map((name) => `C:\\proj\\${name}.txt`)
      .join("\n");
    const result = parseWindowsWorkspaceFileList(stdout, "C:\\proj", options);
    expect(result.paths).toEqual(["a.txt", "b.txt", "c.txt"]);
    expect(result.isTruncated).toBe(true);
  });
});
