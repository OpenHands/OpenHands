/**
 * Lock files that every Canvas process sharing a folder respects: several
 * Canvas processes (or the desktop app and the server) may share one
 * DigitalOcean store, and a refresh token works only once.
 *
 * A lock is a file created exclusively, holding who made it: pid, host, PID
 * namespace and a nonce. A lock whose process is gone (on this host, in this
 * PID namespace), or that is older than any holder keeps it, is taken over.
 * A pid from another host or namespace says nothing here (`kill` would not
 * find it), so only age ends such a lock.
 *
 * Taking over is the step that needs care: two processes may both judge the
 * same lock stale. Each must then remove only the lock it inspected, never a
 * fresh one another process created meanwhile. So the removal runs under a
 * second, short-lived lock (`<lock>.break`) and re-reads the lock first. The
 * break lock is held for one read and one unlink; one left by a process that
 * died in that instant is removed by age.
 */

import { randomUUID } from "node:crypto";
import {
  linkSync,
  mkdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const BREAK_LOCK_STALE_MS = 10_000;
const POLL_MS = 25;

/** This process's PID namespace on Linux; null elsewhere. */
function pidNamespace() {
  try {
    return readlinkSync("/proc/self/ns/pid");
  } catch {
    return null;
  }
}

const HOST = hostname();
const NAMESPACE = pidNamespace();

function ownerRecord() {
  return JSON.stringify({
    pid: process.pid,
    host: HOST,
    ns: NAMESPACE,
    nonce: randomUUID(),
  });
}

function readOrNull(file) {
  try {
    return readFileSync(file, "utf8");
  } catch {
    return null;
  }
}

/**
 * Create `file` holding `content`, or report that it exists. The content is
 * written to a file of its own first and linked into place, so a reader never
 * sees a lock without its owner. Where hard links are not supported, the file
 * is created exclusively and written; a reader that sees it empty waits.
 */
function createExclusive(file, content) {
  const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
  writeFileSync(temporary, content, { mode: 0o600 });
  try {
    linkSync(temporary, file);
    return true;
  } catch (error) {
    if (error?.code === "EEXIST") return false;
    try {
      writeFileSync(file, content, { flag: "wx", mode: 0o600 });
      return true;
    } catch (fallback) {
      if (fallback?.code === "EEXIST") return false;
      throw fallback;
    }
  } finally {
    rmSync(temporary, { force: true });
  }
}

function ageMs(file) {
  try {
    return Date.now() - statSync(file).mtimeMs;
  } catch {
    return null;
  }
}

/** Whether the process that wrote `content` is known to be gone. */
function ownerIsGone(content) {
  let owner;
  try {
    owner = JSON.parse(content);
  } catch {
    return false;
  }
  if (!owner?.pid || owner.host !== HOST || owner.ns !== NAMESPACE) {
    return false;
  }
  try {
    process.kill(owner.pid, 0);
    return false;
  } catch (error) {
    return error?.code === "ESRCH";
  }
}

/**
 * Remove the lock at `file` if it is stale. Returns true when the caller
 * should try to take the lock again at once.
 */
function breakIfStale(file, staleMs) {
  const inspected = readOrNull(file);
  if (inspected === null) return true;
  const age = ageMs(file);
  if (age === null) return true;
  if (age <= staleMs && !ownerIsGone(inspected)) return false;

  const breakFile = `${file}.break`;
  if (!createExclusive(breakFile, ownerRecord())) {
    const breakAge = ageMs(breakFile);
    if (breakAge !== null && breakAge > BREAK_LOCK_STALE_MS) {
      rmSync(breakFile, { force: true });
    }
    return false;
  }
  try {
    // Another process may have broken it and made a fresh one meanwhile.
    if (readOrNull(file) === inspected) rmSync(file, { force: true });
  } finally {
    rmSync(breakFile, { force: true });
  }
  return true;
}

function release(file, content) {
  // A lock taken over from this process (only possible past `staleMs`)
  // belongs to its new holder now.
  if (readOrNull(file) === content) rmSync(file, { force: true });
}

/**
 * @typedef {object} LockOptions
 * @property {number} staleMs Longer than any holder keeps the lock
 * @property {number} waitMs How long to wait for it before giving up
 * @property {string} busyMessage The error thrown after `waitMs`
 */

/**
 * Run `fn` (which may be async) holding the lock file at `file`.
 *
 * @template T
 * @param {string} file
 * @param {() => Promise<T> | T} fn
 * @param {LockOptions} options
 * @returns {Promise<T>}
 */
export async function withLockFile(file, fn, { staleMs, waitMs, busyMessage }) {
  mkdirSync(path.dirname(file), { recursive: true });
  const content = ownerRecord();
  const deadline = Date.now() + waitMs;
  while (!createExclusive(file, content)) {
    if (breakIfStale(file, staleMs)) continue;
    if (Date.now() >= deadline) throw new Error(busyMessage);
    await sleep(POLL_MS);
  }
  try {
    return await fn();
  } finally {
    release(file, content);
  }
}

const pause = new Int32Array(new SharedArrayBuffer(4));

/**
 * Run `fn` synchronously holding the lock file at `file`. For short
 * read-and-rewrite steps only: waiting for another process blocks this one.
 *
 * @template T
 * @param {string} file
 * @param {() => T} fn
 * @param {LockOptions} options
 * @returns {T}
 */
export function withLockFileSync(file, fn, { staleMs, waitMs, busyMessage }) {
  mkdirSync(path.dirname(file), { recursive: true });
  const content = ownerRecord();
  const deadline = Date.now() + waitMs;
  while (!createExclusive(file, content)) {
    if (breakIfStale(file, staleMs)) continue;
    if (Date.now() >= deadline) throw new Error(busyMessage);
    Atomics.wait(pause, 0, 0, POLL_MS);
  }
  try {
    return fn();
  } finally {
    release(file, content);
  }
}
