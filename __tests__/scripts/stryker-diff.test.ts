// @vitest-environment node
import { spawnSync } from "node:child_process";
import type { SpawnSyncOptions } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { main } from "../../scripts/stryker-diff.mjs";

const scriptPath = fileURLToPath(
  new URL("../../scripts/stryker-diff.mjs", import.meta.url),
);

interface SpawnResult {
  error?: Error;
  status: number | null;
  stderr: string;
  stdout: string | null;
}

interface ProcessCall {
  args: readonly string[];
  command: string;
  options?: SpawnSyncOptions;
}

function executeWithResults(
  argv: string[],
  results: SpawnResult[],
): { calls: ProcessCall[]; errors: string; output: string; status: number } {
  const calls: ProcessCall[] = [];
  const output: string[] = [];
  const errors: string[] = [];

  const status = main(argv, {
    spawn(
      command: string,
      args: readonly string[],
      options?: SpawnSyncOptions,
    ) {
      calls.push({ command, args, options });
      const result = results[calls.length - 1];
      if (!result) {
        throw new Error(
          `Unexpected process call: ${command} ${args.join(" ")}`,
        );
      }
      return result;
    },
    writeError(message: string) {
      errors.push(message);
    },
    writeOutput(message: string) {
      output.push(message);
    },
  });

  return {
    calls,
    errors: errors.join(""),
    output: output.join(""),
    status,
  };
}

function successfulProcess(stdout = ""): SpawnResult {
  return { status: 0, stderr: "", stdout };
}

describe("Stryker diff runner", () => {
  it("defaults to main and exits successfully when nothing changed", () => {
    const result = executeWithResults([], [successfulProcess()]);

    expect(result.calls).toHaveLength(1);
    expect(result.calls[0]).toMatchObject({
      command: "git",
      args: ["diff", "--name-only", "--diff-filter=ACMRTUXB", "main...HEAD"],
      options: { encoding: "utf8", windowsHide: true },
    });
    expect(result.output).toBe(
      "No changed production files to mutate against main.\n",
    );
    expect(result.status).toBe(0);
  });

  it("uses the requested base ref and mutates changed TypeScript production files", () => {
    const result = executeWithResults(
      ["origin/main"],
      [successfulProcess("src/a.ts\nsrc/ui/card.tsx\n"), successfulProcess()],
    );

    expect(result.calls[0]?.args).toContain("origin/main...HEAD");
    expect(result.calls[1]?.args.slice(1)).toEqual([
      "run",
      "--incremental",
      "--force",
      "--mutate",
      "src/a.ts,src/ui/card.tsx",
    ]);
    expect(result.calls[1]?.args[0]).toMatch(
      /node_modules\/@stryker-mutator\/core\/bin\/stryker\.js$/,
    );
    expect(result.calls[1]?.options).toEqual({
      stdio: "inherit",
      windowsHide: true,
    });
    expect(result.status).toBe(0);
  });

  it("filters non-production and non-TypeScript paths before mutation", () => {
    const changedFiles = [
      "src/keep.ts",
      "src/keep.tsx",
      "src/a.test.ts.helper.ts",
      "src\\windows\\keep.ts",
      "src/keep.js",
      "src/looks-like.ts.backup",
      "nested/src/not-rooted.ts",
      "scripts/tool.mjs",
      "tools/canvas_ui_tool.py",
      "src/a.test.ts",
      "src/a.spec.tsx",
      "src/types/a.d.ts",
      "src/fixtures/a.ts",
      "src/mocks/a.ts",
      "src/__tests__/a.ts",
      "src/test/a.ts",
      "src/x/__snapshots__/a.ts",
      "src/generated/a.ts",
      "src/a.gen.ts",
      "src/a.generated.tsx",
      "build/a.ts",
    ].join("\n");

    const result = executeWithResults(
      [],
      [successfulProcess(`${changedFiles}\n`), successfulProcess()],
    );

    expect(result.calls[1]?.args.at(-1)).toBe(
      "src/keep.ts,src/keep.tsx,src/a.test.ts.helper.ts,src\\windows\\keep.ts",
    );
  });

  it("passes paths with spaces as one mutation argument", () => {
    const result = executeWithResults(
      [],
      [
        successfulProcess("src/path with spaces/a.ts\nsrc/b.tsx\n"),
        successfulProcess(),
      ],
    );

    const mutationArgs = result.calls[1]?.args ?? [];
    expect(mutationArgs.at(-2)).toBe("--mutate");
    expect(mutationArgs.at(-1)).toBe("src/path with spaces/a.ts,src/b.tsx");
  });

  it("does not run Stryker when every changed file is filtered out", () => {
    const result = executeWithResults(
      [],
      [
        successfulProcess(
          "README.md\nsrc/a.test.ts\ntools/canvas_ui_tool.py\nscripts/a.mjs\n",
        ),
      ],
    );

    expect(result.calls).toHaveLength(1);
    expect(result.output).toContain("No changed production files");
    expect(result.status).toBe(0);
  });

  it("propagates Stryker's failing exit status", () => {
    const result = executeWithResults(
      [],
      [
        successfulProcess("src/a.ts\n"),
        { status: 23, stdout: "", stderr: "mutation failed" },
      ],
    );

    expect(result.status).toBe(23);
  });

  it("reports git diff failures without starting Stryker", () => {
    const result = executeWithResults(
      ["missing-ref"],
      [
        {
          status: 128,
          stdout: "",
          stderr: "  fatal: ambiguous argument  ",
        },
      ],
    );

    expect(result.calls).toHaveLength(1);
    expect(result.errors).toBe(
      "Unable to determine changed files against missing-ref: fatal: ambiguous argument\n",
    );
    expect(result.status).toBe(128);
  });

  it("reports an unknown git error when no diagnostic is available", () => {
    const result = executeWithResults(
      [],
      [{ status: 128, stdout: "", stderr: "" }],
    );

    expect(result.errors).toBe(
      "Unable to determine changed files against main: unknown error\n",
    );
    expect(result.status).toBe(128);
  });

  it("reports git process errors when no exit status is available", () => {
    const result = executeWithResults(
      [],
      [
        {
          error: new Error("git executable not found"),
          status: null,
          stdout: "",
          stderr: "",
        },
      ],
    );

    expect(result.calls).toHaveLength(1);
    expect(result.errors).toBe(
      "Unable to determine changed files against main: git executable not found\n",
    );
    expect(result.status).toBe(1);
  });

  it("reports Stryker process errors", () => {
    const result = executeWithResults(
      [],
      [
        successfulProcess("src/a.ts\n"),
        {
          error: new Error("Stryker executable failed to start"),
          status: null,
          stdout: "",
          stderr: "",
        },
      ],
    );

    expect(result.errors).toBe(
      "Unable to start Stryker: Stryker executable failed to start\n",
    );
    expect(result.status).toBe(1);
  });

  it("fails when Stryker exits without a status", () => {
    const result = executeWithResults(
      [],
      [
        successfulProcess("src/a.ts\n"),
        { status: null, stdout: "", stderr: "" },
      ],
    );

    expect(result.status).toBe(1);
  });

  it("runs as a CLI in a clean git repository without invoking Stryker", () => {
    const repo = mkdtempSync(path.join(tmpdir(), "stryker-diff-"));

    try {
      expect(
        spawnSync("git", ["init", "-b", "main"], {
          cwd: repo,
          encoding: "utf8",
        }).status,
      ).toBe(0);
      writeFileSync(path.join(repo, "README.md"), "# fixture\n");
      expect(
        spawnSync("git", ["add", "README.md"], {
          cwd: repo,
          encoding: "utf8",
        }).status,
      ).toBe(0);
      expect(
        spawnSync(
          "git",
          [
            "-c",
            "user.name=Stryker Test",
            "-c",
            "user.email=stryker@example.test",
            "commit",
            "-m",
            "fixture",
          ],
          { cwd: repo, encoding: "utf8" },
        ).status,
      ).toBe(0);

      const result = spawnSync(process.execPath, [scriptPath], {
        cwd: repo,
        encoding: "utf8",
      });

      expect(result.stderr).toBe("");
      expect(result.stdout).toBe(
        "No changed production files to mutate against main.\n",
      );
      expect(result.status).toBe(0);
    } finally {
      rmSync(repo, { force: true, recursive: true });
    }
  });
});
