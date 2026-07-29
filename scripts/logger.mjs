/**
 * Shared file logger for agent-canvas dev scripts.
 *
 * Writes log output to a daily-rotating file under
 * <state-dir>/logs/agent-canvas.YYYY-MM-DD.log (7-day retention) alongside
 * the existing console output (which is unchanged).
 *
 * winston is loaded dynamically and treated as optional: when it isn't
 * resolvable — most importantly inside the packaged Electron desktop app,
 * whose `afterPack` hook strips `Resources/app/node_modules/` — `fileLog`
 * becomes a no-op and console logging continues to work unchanged. See
 * AGENTS.md "Electron desktop packaging" for the strip-hook details.
 */

import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import process from "node:process";

// Mirror the state-directory logic from dev-safe.mjs so log files live
// alongside all other agent-canvas runtime state (e.g. ~/.openhands/agent-canvas).
// The same env var (OH_CANVAS_SAFE_STATE_DIR) overrides both.
const stateDir =
  process.env.OH_CANVAS_SAFE_STATE_DIR ||
  join(homedir(), ".openhands", "agent-canvas");
const logDir = join(stateDir, "logs");

// Matches any ANSI CSI escape sequence (colors, cursor movement, etc.).
const ANSI_RE = /\x1b\[[0-9;]*m/g;

/**
 * Remove ANSI escape codes so log files contain clean plain text.
 * @param {string} str
 * @returns {string}
 */
export function stripAnsi(str) {
  return typeof str === "string" ? str.replace(ANSI_RE, "") : String(str);
}

/**
 * Attempt to construct a winston-backed file logger. Returns `null` if
 * winston isn't installed (packaged desktop app) or any setup step fails;
 * `fileLog` then degrades to a no-op.
 *
 * @returns {Promise<import("winston").Logger | null>}
 */
async function createFileLogger() {
  let winston;
  let DailyRotateFileMod;
  try {
    winston = await import("winston");
    DailyRotateFileMod = await import("winston-daily-rotate-file");
  } catch {
    return null;
  }

  try {
    mkdirSync(logDir, { recursive: true });

    const DailyRotateFile =
      DailyRotateFileMod.default ?? DailyRotateFileMod;
    const fileTransport = new DailyRotateFile({
      dirname: logDir,
      filename: "agent-canvas.%DATE%.log",
      datePattern: "YYYY-MM-DD",
      maxFiles: "7d",
      auditFile: join(logDir, ".log-audit.json"),
      createSymlink: false,
    });

    const logger = winston.createLogger({
      level: "debug",
      format: winston.format.combine(
        winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
        winston.format.printf(
          ({ timestamp, level, message }) =>
            `${timestamp} [${level.toUpperCase().padEnd(5)}] ${message}`,
        ),
      ),
      transports: [fileTransport],
    });

    // Swallow any transport-level errors (e.g. disk full) so a logging
    // failure never crashes the dev server.
    logger.on("error", () => {});
    fileTransport.on("error", () => {});
    return logger;
  } catch {
    return null;
  }
}

const fileLogger = await createFileLogger();

/**
 * Write a message to the rotating log file. No-ops when winston isn't
 * available (e.g. the packaged desktop app). ANSI escape codes are
 * stripped automatically; console output is unaffected.
 *
 * @param {'info' | 'warn' | 'error' | 'debug'} level
 * @param {string} message
 */
export function fileLog(level, message) {
  if (!fileLogger) return;
  fileLogger.log(level, stripAnsi(message));
}
