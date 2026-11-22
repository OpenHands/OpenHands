// @vitest-environment node
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  parseChecksums,
  verifyChecksum,
} from "../../scripts/download-integrity.mjs";

const DIGEST_A = createHash("sha256").update("payload-a").digest("hex");
const DIGEST_B = createHash("sha256").update("payload-b").digest("hex");

describe("parseChecksums", () => {
  it("parses the SHASUMS256.txt / uv *.sha256 entry format", () => {
    const entries = parseChecksums(
      `${DIGEST_A}  node-v24.22.1-linux-x64.tar.gz\n` +
        `${DIGEST_B}  uv-x86_64-pc-windows-msvc.zip\n`,
      "SHASUMS256.txt",
    );
    expect(entries.get("node-v24.22.1-linux-x64.tar.gz")).toBe(DIGEST_A);
    expect(entries.get("uv-x86_64-pc-windows-msvc.zip")).toBe(DIGEST_B);
  });

  it("fails closed on a malformed line instead of skipping it", () => {
    expect(() =>
      parseChecksums("not-a-digest  file.zip\n", "uv release .sha256"),
    ).toThrow(/Malformed checksum line/);
  });
});

describe("verifyChecksum", () => {
  function makeArchive(content: string): string {
    const dir = mkdtempSync(join(tmpdir(), "download-integrity-"));
    const file = join(dir, "archive.tar.gz");
    writeFileSync(file, content);
    return file;
  }

  it("passes when the downloaded archive matches its publisher entry", () => {
    const file = makeArchive("payload-a");
    expect(
      verifyChecksum(
        file,
        "archive.tar.gz",
        `${DIGEST_A}  archive.tar.gz\n`,
        "SHASUMS256.txt",
      ),
    ).toBe(DIGEST_A);
  });

  it("fails closed on a mismatch, naming the artifact", () => {
    const file = makeArchive("tampered");
    expect(() =>
      verifyChecksum(
        file,
        "archive.tar.gz",
        `${DIGEST_A}  archive.tar.gz\n`,
        "SHASUMS256.txt",
      ),
    ).toThrow(/Checksum mismatch for "archive\.tar\.gz"/);
  });

  it("fails closed when the checksum file has no entry for the artifact", () => {
    const file = makeArchive("payload-a");
    expect(() =>
      verifyChecksum(
        file,
        "archive.tar.gz",
        `${DIGEST_A}  other-file.zip\n`,
        "SHASUMS256.txt",
      ),
    ).toThrow(/No checksum entry for "archive\.tar\.gz"/);
  });
});
