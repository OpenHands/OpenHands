// @vitest-environment node
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { withLockFile } from "../../scripts/mars-lock-file.mjs";

const LOCK_MODULE = new URL("../../scripts/mars-lock-file.mjs", import.meta.url)
  .href;
const OPTIONS = { staleMs: 60_000, waitMs: 20_000, busyMessage: "busy" };

/** A lock file left long ago, which any process may take over. */
function leaveStaleLock(lockPath: string) {
  writeFileSync(lockPath, "left by a process long gone");
  const longAgo = new Date(Date.now() - 10 * 60_000);
  utimesSync(lockPath, longAgo, longAgo);
}

function runWorkers(script: string, args: string[][]) {
  return Promise.all(
    args.map(
      (workerArgs) =>
        new Promise<void>((done, fail) => {
          const child = spawn(process.execPath, [script, ...workerArgs], {
            stdio: ["ignore", "ignore", "pipe"],
          });
          let stderr = "";
          child.stderr.on("data", (chunk) => {
            stderr += chunk;
          });
          child.on("exit", (code) =>
            code === 0 ? done() : fail(new Error(stderr || `exit ${code}`)),
          );
        }),
    ),
  );
}

describe("withLockFile", () => {
  it("lets one process at a time take a lock found stale, however many find it at once", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mars-lock-"));
    const lockPath = join(dir, "grant.lock");
    const logPath = join(dir, "holders.log");
    const script = join(dir, "worker.mjs");
    // "abandon" takes the lock and exits holding it; the others start at the
    // same moment, find it stale together, and log while they hold it.
    writeFileSync(
      script,
      `import { appendFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { withLockFile } from ${JSON.stringify(LOCK_MODULE)};
const [lockPath, logPath, startAt] = process.argv.slice(2);
if (logPath === "abandon") {
  await withLockFile(lockPath, () => process.exit(0), ${JSON.stringify(OPTIONS)});
}
while (Date.now() < Number(startAt));
await withLockFile(lockPath, async () => {
  appendFileSync(logPath, "in\\n");
  await sleep(20);
  appendFileSync(logPath, "out\\n");
}, ${JSON.stringify(OPTIONS)});
`,
    );

    for (let round = 0; round < 8; round += 1) {
      await runWorkers(script, [[lockPath, "abandon"]]);
      const startAt = String(Date.now() + 400);
      await runWorkers(
        script,
        Array.from({ length: 4 }, () => [lockPath, logPath, startAt]),
      );
    }

    // Holders never overlap: every "in" is followed by its own "out".
    const log = readFileSync(logPath, "utf8").trim().split("\n");
    expect(log).toHaveLength(64);
    expect(log.join(" ")).toBe(Array(32).fill("in out").join(" "));
  }, 60_000);

  it("waits out a lock from another host or PID namespace until it is old", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mars-lock-"));
    const lockPath = join(dir, "grant.lock");
    // Its pid means nothing here: that process may well be alive.
    writeFileSync(
      lockPath,
      JSON.stringify({ pid: 999_999, host: "another-host", ns: null }),
    );

    await expect(
      withLockFile(lockPath, () => "ran", { ...OPTIONS, waitMs: 200 }),
    ).rejects.toThrow("busy");

    leaveStaleLock(lockPath);
    await expect(withLockFile(lockPath, () => "ran", OPTIONS)).resolves.toBe(
      "ran",
    );
  });

  it("leaves a lock another process took over from this one", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mars-lock-"));
    const lockPath = join(dir, "grant.lock");

    await withLockFile(
      lockPath,
      () => writeFileSync(lockPath, "the new holder"),
      OPTIONS,
    );

    expect(readFileSync(lockPath, "utf8")).toBe("the new holder");
  });
});
