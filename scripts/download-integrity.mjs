/**
 * Checksum verification for the desktop build's downloaded runtimes
 * (the Node distribution and the uv release archives).
 *
 * Both publishers ship the same checksum format — "<sha256>  <filename>"
 * (hex, two spaces, name): Node as one SHASUMS256.txt per dist directory,
 * uv as a per-asset "<archive>.sha256" companion. A downloaded archive
 * whose digest does not match the publisher's entry aborts the build
 * before anything is bundled.
 *
 * Pure — no network, no filesystem writes beyond hashing the file the
 * caller points at — so it is unit tested offline in
 * __tests__/scripts/download-integrity.test.ts.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

/**
 * Parse "<sha256>  <filename>" lines (SHASUMS256.txt and uv *.sha256) into
 * a filename → digest map. Fail closed on a malformed line: a checksum
 * file we cannot parse fully is not a checksum file we can trust.
 *
 * @param {string} text The checksum file's contents.
 * @param {string} source Human-readable name for error messages.
 * @returns {Map<string, string>} filename → lowercase hex sha256.
 */
export function parseChecksums(text, source) {
  const entries = new Map();
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = /^([0-9a-fA-F]{64})\s+\*?(.+)$/.exec(line);
    if (!match) {
      throw new Error(
        `Malformed checksum line in ${source}: ${JSON.stringify(rawLine)}`,
      );
    }
    entries.set(match[2], match[1].toLowerCase());
  }
  return entries;
}

/** sha256 of a file on disk, hex-encoded. */
function sha256File(filePath) {
  return createHash("sha256")
    .update(readFileSync(filePath))
    .digest("hex");
}

/**
 * Verify the file at `filePath` matches the publisher's checksum entry for
 * `filename`, and return the actual digest. Fails closed — a mismatch or a
 * missing entry names the artifact and aborts the build before packaging.
 *
 * @param {string} filePath Downloaded archive on disk.
 * @param {string} filename Artifact name as listed in the checksum file.
 * @param {string} checksumText Contents of the publisher's checksum file.
 * @param {string} source Human-readable name for error messages.
 * @returns {string} The verified lowercase hex sha256.
 */
export function verifyChecksum(filePath, filename, checksumText, source) {
  const entries = parseChecksums(checksumText, source);
  const expected = entries.get(filename);
  if (!expected) {
    throw new Error(
      `No checksum entry for "${filename}" in ${source}. The publisher's ` +
        "checksum file may have changed — refusing to bundle an unverified download.",
    );
  }
  const actual = sha256File(filePath);
  if (actual !== expected) {
    throw new Error(
      `Checksum mismatch for "${filename}" (from ${source}): ` +
        `expected ${expected}, got ${actual}. The download failed or was ` +
        "tampered with — aborting before packaging.",
    );
  }
  return actual;
}
