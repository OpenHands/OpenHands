import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  resetAutomationMigrationLatch,
  watchAutomationMigration,
} from "../../scripts/dev-process-utils.mjs";

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

function fireMigrationExit(proc: FakeProc) {
  proc.stdout.emit("data", Buffer.from(MIGRATION_LOG_LINE));
  proc.emit("exit", 3);
}

const MIGRATION_LOG_LINE =
  'alembic.runtime.migration: Can\'t locate revision identified by "abc123"';

describe("watchAutomationMigration", () => {
  beforeEach(() => {
    resetAutomationMigrationLatch();
  });

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

    fireMigrationExit(proc);

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
    fireMigrationExit(proc);

    // Second failure on the same process: no second recovery, we give up.
    fireMigrationExit(proc);

    expect(recoverCount).toBe(1);
    expect(logs.some((m) => m.includes("giving up"))).toBe(true);
  });

  it("does not recover again on a new watcher after a restart", () => {
    // This is the real failure mode: recovery respawns the backend, which
    // installs a *new* watcher. A per-instance latch would miss this and
    // loop forever.
    const first = makeFakeProc();
    let recoverCount = 0;
    const logs: string[] = [];

    watchAutomationMigration(first, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recoverCount += 1;
      },
      log: (message: string) => logs.push(message),
    });
    fireMigrationExit(first);
    expect(recoverCount).toBe(1);

    const second = makeFakeProc();
    watchAutomationMigration(second, {
      dbPath: "/tmp/fake-automations.db",
      onRecover: () => {
        recoverCount += 1;
      },
      log: (message: string) => logs.push(message),
    });
    fireMigrationExit(second);

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

    expect(() => fireMigrationExit(proc)).not.toThrow();
    expect(
      logs.some((m) => m.includes("Failed to remove stale automations.db")),
    ).toBe(true);
  });
});

// Real-filesystem integration: prove onRecover wiring deletes an actual file.
describe("watchAutomationMigration integration", () => {
  let dir: string;

  beforeEach(() => {
    resetAutomationMigrationLatch();
  });

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

    fireMigrationExit(proc);
    expect(existsSync(dbPath)).toBe(false);

    // Second failure must not attempt anything further.
    proc.emit("exit", 3);
    expect(existsSync(dbPath)).toBe(false);
  });
});
