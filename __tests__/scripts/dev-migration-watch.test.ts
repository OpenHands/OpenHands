import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { watchAutomationMigration } from "../../scripts/dev-process-utils.mjs";

type LogKind = "warn" | "ok" | "error";

interface FakeProc extends EventEmitter {
  stdout: EventEmitter;
  stderr: EventEmitter;
}

/** Minimal fake of a spawned child: EventEmitters for stdout/stderr/self. */
function makeFakeProc(): FakeProc {
  const proc = new EventEmitter() as FakeProc;
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  return proc;
}

const MIGRATION_LOG_LINE =
  'alembic.runtime.migration: Can\'t locate revision identified by "abc123"';

describe("watchAutomationMigration", () => {
  it("recovers (deletes the DB) when the migration error and exit code 3 line up", () => {
    const proc = makeFakeProc();
    let recovered = false;
    watchAutomationMigration(proc, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recovered = true;
      },
      log: () => {},
    });

    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
    proc.emit("exit", 3);

    expect(recovered).toBe(true);
  });

  it("does not delete the DB when the exit code is not 3", () => {
    const proc = makeFakeProc();
    let recovered = false;
    watchAutomationMigration(proc, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recovered = true;
      },
      log: () => {},
    });

    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
    proc.emit("exit", 1);

    expect(recovered).toBe(false);
  });

  it("does not delete the DB when the log pattern never appeared", () => {
    const proc = makeFakeProc();
    let recovered = false;
    watchAutomationMigration(proc, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recovered = true;
      },
      log: () => {},
    });

    // A coincidental exit 3 from some other cause must never delete the DB.
    proc.emit("exit", 3);

    expect(recovered).toBe(false);
  });

  it("latches after one recovery attempt so a broken backend cannot loop", () => {
    const proc = makeFakeProc();
    let recoverCount = 0;
    const logs: string[] = [];
    watchAutomationMigration(proc, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recoverCount += 1;
      },
      log: (message: string) => logs.push(message),
    });

    // First failure: recovery runs.
    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
    proc.emit("exit", 3);

    // Second failure (restart failed too): no second recovery, we give up.
    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
    proc.emit("exit", 3);

    expect(recoverCount).toBe(1);
    expect(logs.some((m) => m.includes("giving up"))).toBe(true);
  });

  it("reports a recovery failure instead of throwing", () => {
    const proc = makeFakeProc();
    const logs: string[] = [];
    watchAutomationMigration(proc, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        throw new Error("EACCES: read-only file system");
      },
      log: (message: string) => logs.push(message),
    });

    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));

    expect(() => proc.emit("exit", 3)).not.toThrow();
    expect(
      logs.some((m) => m.includes("Failed to remove stale automations.db")),
    ).toBe(true);
  });
});

// Real-filesystem integration: prove onRecover wiring deletes an actual file.
describe("watchAutomationMigration integration", () => {
  let dir: string;

  afterEach(() => {
    if (dir && existsSync(dir)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("deletes a real stale DB file exactly once across repeated failures", () => {
    dir = mkdtempSync(join(tmpdir(), "oh-migration-test-"));
    const dbPath = join(dir, "automations.db");
    writeFileSync(dbPath, "stale sqlite data");

    const proc = makeFakeProc();
    watchAutomationMigration(proc, {
      dbPath,
      onRecover: () => rmSync(dbPath),
      log: () => {},
    });

    proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
    proc.emit("exit", 3);
    expect(existsSync(dbPath)).toBe(false);

    // Second failure must not attempt anything further.
    proc.emit("exit", 3);
    expect(existsSync(dbPath)).toBe(false);
  });
});
