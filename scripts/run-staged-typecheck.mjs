#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";

function getNpmInvocation() {
  if (process.platform !== "win32") {
    return { command: "npm", args: ["run", "typecheck:staged"] };
  }

  const npmCliPath =
    process.env.npm_execpath ??
    join(dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js");

  if (existsSync(npmCliPath)) {
    return {
      command: process.execPath,
      args: [npmCliPath, "run", "typecheck:staged"],
    };
  }

  return {
    command: process.env.ComSpec ?? "cmd.exe",
    args: ["/d", "/s", "/c", "npm run typecheck:staged"],
  };
}

const { command, args } = getNpmInvocation();

const result = spawnSync(command, args, {
  stdio: "inherit",
  windowsHide: true,
});

if (result.error) {
  throw result.error;
}

if (result.signal) {
  console.error(`typecheck:staged terminated by signal ${result.signal}`);
  process.exit(1);
}

process.exit(result.status ?? 1);
