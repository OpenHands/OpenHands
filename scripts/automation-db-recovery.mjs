/**
 * Stale automation SQLite DB recovery.
 *
 * When `openhands-automation` is upgraded to a version that restructured its
 * Alembic migration history, the on-disk `automations.db` can point at a
 * revision that no longer exists in the new migration chain. Alembic's
 * `upgrade head` then fails on startup with e.g.:
 *
 *   Failed to apply SQLite migrations: Can't locate revision identified by '007'
 *
 * The automation backend exits non-zero and the Automate tab becomes
 * unavailable. This module gives the dev launchers a narrowly-scoped recovery:
 * detect that specific failure in the service output, delete the stale DB
 * (plus its SQLite sidecar files) and restart the backend exactly once.
 *
 * Kept dependency-free and side-effect-injectable so the detection/retry
 * behaviour can be unit tested without spawning real processes.
 */

import { existsSync, rmSync } from "node:fs";

// Patterns that identify a stale-migration startup failure. Kept intentionally
// specific so unrelated automation crashes are never silently masked.
export const AUTOMATION_MIGRATION_ERROR_PATTERNS = [
  /Failed to apply SQLite migrations/i,
  /Can't locate revision identified by/i,
];

/**
 * @param {string} line A single line of automation service output.
 * @returns {boolean} True if the line matches a stale-migration failure.
 */
export function isAutomationMigrationError(line) {
  if (!line) return false;
  return AUTOMATION_MIGRATION_ERROR_PATTERNS.some((pattern) =>
    pattern.test(line),
  );
}

/**
 * SQLite keeps auxiliary files next to the main DB (write-ahead log, shared
 * memory, rollback journal). A clean reset removes all of them.
 *
 * @param {string} dbPath Absolute path to the `automations.db` file.
 * @returns {string[]} The DB path plus its sidecar files.
 */
export function automationDbSidecarPaths(dbPath) {
  return [dbPath, `${dbPath}-wal`, `${dbPath}-shm`, `${dbPath}-journal`];
}

/**
 * Create a recovery helper for a single automation backend launch.
 *
 * The returned `handleLine`/`handleExit` callbacks are meant to be wired into
 * the launcher's `spawnService` for the automation service. Recovery fires at
 * most once per helper instance: the caller is responsible for arming a fresh
 * helper only on the first launch attempt so a persistently-broken DB cannot
 * cause an infinite restart loop.
 *
 * @param {object} options
 * @param {string} options.dbPath Absolute path to the automation SQLite DB.
 * @param {(message: string) => void} options.log Logger for recovery messages.
 * @param {() => void} options.restart Re-launches the automation backend.
 * @param {{ existsSync: Function, rmSync: Function }} [options.fs] Injected fs.
 * @returns {{ handleLine: (line: string) => void, handleExit: (code: number|null) => boolean }}
 */
export function createAutomationDbRecovery({ dbPath, log, restart, fs }) {
  const fileSystem = fs ?? { existsSync, rmSync };
  let sawMigrationError = false;
  let recovered = false;

  return {
    handleLine(line) {
      if (!sawMigrationError && isAutomationMigrationError(line)) {
        sawMigrationError = true;
      }
    },

    /**
     * @returns {boolean} True when a recovery restart was triggered, so the
     *   caller can suppress the generic "exited with code N" error log.
     */
    handleExit(code) {
      // A clean exit or an already-consumed recovery attempt is not our case.
      if (code === 0 || code === null) return false;
      if (recovered || !sawMigrationError) return false;

      recovered = true;
      log(
        `Detected a stale automation database (SQLite migration failure). ` +
          `Resetting ${dbPath} and restarting the automation service once...`,
      );

      let removedAny = false;
      for (const path of automationDbSidecarPaths(dbPath)) {
        try {
          if (fileSystem.existsSync(path)) {
            fileSystem.rmSync(path);
            removedAny = true;
          }
        } catch (error) {
          log(
            `Failed to delete stale automation DB file ${path}: ${error.message}`,
          );
          return false;
        }
      }

      if (!removedAny) {
        log(
          `No automation database file found at ${dbPath}; skipping automatic ` +
            `reset. The migration error is likely unrelated to a stale local DB.`,
        );
        return false;
      }

      restart();
      return true;
    },
  };
}
