#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const EXCLUDED_DIRECTORY =
  /(^|\/)(__tests__|tests?|fixtures|__fixtures__|snapshots|__snapshots__|generated|mocks|dev)(\/|$)/;
const EXCLUDED_FILE = /\.(test|spec|d|gen|generated)\.(ts|tsx)$/;

function isProductionTypeScript(file) {
  const normalized = file.replaceAll("\\", "/");
  return (
    /^src\/.+\.(ts|tsx)$/.test(normalized) &&
    !EXCLUDED_DIRECTORY.test(normalized) &&
    !EXCLUDED_FILE.test(normalized)
  );
}

/**
 * @typedef {object} ProcessResult
 * @property {Error | undefined} [error]
 * @property {number | null} status
 * @property {string | null} stderr
 * @property {string | null} stdout
 */

/**
 * @typedef {object} Dependencies
 * @property {(command: string, args: readonly string[], options?: import("node:child_process").SpawnSyncOptions) => ProcessResult} spawn
 * @property {(message: string) => void} writeError
 * @property {(message: string) => void} writeOutput
 */

// Stryker disable all: the real process adapter is verified by the CLI
// subprocess smoke test, which Vitest cannot attribute to the parent process.
/** @type {Dependencies} */
const defaultDependencies = {
  spawn(command, args, options) {
    const result = spawnSync(command, [...args], options);
    return {
      error: result.error,
      status: result.status,
      stderr: result.stderr?.toString() ?? null,
      stdout: result.stdout?.toString() ?? null,
    };
  },
  writeError(message) {
    process.stderr.write(message);
  },
  writeOutput(message) {
    process.stdout.write(message);
  },
};
// Stryker restore all

/**
 * Run mutation testing against production TypeScript files changed from a base
 * git ref. The dependency parameter keeps the process boundary observable in
 * tests while the command-line behavior remains the public API.
 *
 * @param {string[]} argv
 * @param {Partial<Dependencies>} [overrides]
 * @returns {number}
 */
export function main(argv, overrides = {}) {
  const dependencies = { ...defaultDependencies, ...overrides };
  const baseRef = argv[0] ?? "main";
  const gitArgs = [
    "diff",
    "--name-only",
    "--diff-filter=ACMRTUXB",
    `${baseRef}...HEAD`,
  ];
  const gitResult = dependencies.spawn("git", gitArgs, {
    encoding: "utf8",
    windowsHide: true,
  });

  if (gitResult.error || gitResult.status !== 0) {
    const detail =
      gitResult.stderr || gitResult.error?.message || "unknown error";
    dependencies.writeError(
      `Unable to determine changed files against ${baseRef}: ${detail.trim()}\n`,
    );
    return gitResult.status ?? 1;
  }

  const mutationTargets = gitResult.stdout
    .split(/\r?\n/u)
    .filter(isProductionTypeScript);

  if (mutationTargets.length === 0) {
    dependencies.writeOutput(
      `No changed production files to mutate against ${baseRef}.\n`,
    );
    return 0;
  }

  const strykerCliPath = fileURLToPath(
    new URL(
      "../node_modules/@stryker-mutator/core/bin/stryker.js",
      import.meta.url,
    ),
  );
  const strykerResult = dependencies.spawn(
    process.execPath,
    [
      strykerCliPath,
      "run",
      "--incremental",
      "--force",
      "--mutate",
      mutationTargets.join(","),
    ],
    { stdio: "inherit", windowsHide: true },
  );

  if (strykerResult.error) {
    dependencies.writeError(
      `Unable to start Stryker: ${strykerResult.error.message}\n`,
    );
    return 1;
  }

  return strykerResult.status ?? 1;
}

// Stryker disable all: the CLI guard is verified by a real subprocess test,
// which the Vitest runner cannot attribute to the mutated parent process.
const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  process.exitCode = main(process.argv.slice(2));
}
// Stryker restore all
