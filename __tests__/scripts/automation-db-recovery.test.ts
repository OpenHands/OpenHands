// @vitest-environment node
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  automationDbSidecarPaths,
  createAutomationDbRecovery,
  isAutomationMigrationError,
} from "../../scripts/automation-db-recovery.mjs";

const MIGRATION_LINE =
  "Failed to apply SQLite migrations: Can't locate revision identified by '007'";

describe("isAutomationMigrationError", () => {
  it("matches the stale-migration failure lines", () => {
    expect(isAutomationMigrationError(MIGRATION_LINE)).toBe(true);
    expect(
      isAutomationMigrationError("Can't locate revision identified by 'abc'"),
    ).toBe(true);
  });

  it("ignores unrelated automation output", () => {
    expect(isAutomationMigrationError("Application startup complete")).toBe(
      false,
    );
    expect(isAutomationMigrationError("")).toBe(false);
  });
});

describe("automationDbSidecarPaths", () => {
  it("expands the DB path to its SQLite sidecar files", () => {
    expect(automationDbSidecarPaths("/state/automations.db")).toEqual([
      "/state/automations.db",
      "/state/automations.db-wal",
      "/state/automations.db-shm",
      "/state/automations.db-journal",
    ]);
  });
});

describe("createAutomationDbRecovery", () => {
  const tempDirs = [];

  afterEach(() => {
    while (tempDirs.length) {
      rmSync(tempDirs.pop(), { recursive: true, force: true });
    }
  });

  function makeTempDb() {
    const dir = mkdtempSync(path.join(tmpdir(), "automation-recovery-"));
    tempDirs.push(dir);
    const dbPath = path.join(dir, "automations.db");
    writeFileSync(dbPath, "stale");
    writeFileSync(`${dbPath}-wal`, "stale-wal");
    return dbPath;
  }

  it("resets the stale DB and restarts once when a migration error precedes a non-zero exit", () => {
    // Arrange
    const dbPath = makeTempDb();
    const messages = [];
    let restarts = 0;
    const recovery = createAutomationDbRecovery({
      dbPath,
      log: (message) => messages.push(message),
      restart: () => {
        restarts += 1;
      },
    });

    // Act
    recovery.handleLine(MIGRATION_LINE);
    const recovered = recovery.handleExit(3);

    // Assert
    expect(recovered).toBe(true);
    expect(restarts).toBe(1);
    expect(existsSync(dbPath)).toBe(false);
    expect(existsSync(`${dbPath}-wal`)).toBe(false);
    expect(messages.join("\n")).toContain("stale automation database");
  });

  it("recovers at most once so a persistently broken DB cannot loop", () => {
    // Arrange
    const dbPath = makeTempDb();
    let restarts = 0;
    const recovery = createAutomationDbRecovery({
      dbPath,
      log: () => {},
      restart: () => {
        restarts += 1;
      },
    });

    // Act: a second migration failure after the first recovery must be ignored.
    recovery.handleLine(MIGRATION_LINE);
    expect(recovery.handleExit(3)).toBe(true);
    recovery.handleLine(MIGRATION_LINE);
    const secondRecovered = recovery.handleExit(3);

    // Assert
    expect(secondRecovered).toBe(false);
    expect(restarts).toBe(1);
  });

  it("does not touch the DB when the crash is unrelated to a migration", () => {
    // Arrange
    const dbPath = makeTempDb();
    let restarts = 0;
    const recovery = createAutomationDbRecovery({
      dbPath,
      log: () => {},
      restart: () => {
        restarts += 1;
      },
    });

    // Act
    recovery.handleLine("ValueError: something else went wrong");
    const recovered = recovery.handleExit(3);

    // Assert
    expect(recovered).toBe(false);
    expect(restarts).toBe(0);
    expect(existsSync(dbPath)).toBe(true);
  });

  it("does not recover on a clean exit even after a migration warning", () => {
    const dbPath = makeTempDb();
    let restarts = 0;
    const recovery = createAutomationDbRecovery({
      dbPath,
      log: () => {},
      restart: () => {
        restarts += 1;
      },
    });

    recovery.handleLine(MIGRATION_LINE);

    expect(recovery.handleExit(0)).toBe(false);
    expect(recovery.handleExit(null)).toBe(false);
    expect(restarts).toBe(0);
    expect(existsSync(dbPath)).toBe(true);
  });

  it("skips reset and does not restart when no DB file exists to delete", () => {
    // Arrange: injected fs reports the DB as absent.
    const removed = [];
    let restarts = 0;
    const recovery = createAutomationDbRecovery({
      dbPath: "/nonexistent/automations.db",
      log: () => {},
      restart: () => {
        restarts += 1;
      },
      fs: {
        existsSync: () => false,
        rmSync: (p) => removed.push(p),
      },
    });

    // Act
    recovery.handleLine(MIGRATION_LINE);
    const recovered = recovery.handleExit(3);

    // Assert
    expect(recovered).toBe(false);
    expect(restarts).toBe(0);
    expect(removed).toEqual([]);
  });
});
