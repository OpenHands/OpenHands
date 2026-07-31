// @vitest-environment node
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  checksumForFile,
  verifyFileSha256,
} from "../../scripts/download-integrity.mjs";

const SHA256_A = "a".repeat(64);
const SHA256_B = "b".repeat(64);
const TRUSTED_ARCHIVE_SHA256 =
  "1bfefc79b77185639257480b7bd91ec719430531d457075317d5e10dcb655f1d";

let tempDir: string | undefined;

afterEach(async () => {
  if (tempDir) {
    await rm(tempDir, { recursive: true, force: true });
    tempDir = undefined;
  }
});

describe("download integrity", () => {
  it("selects the exact archive from uv and Node checksum formats", () => {
    expect(
      checksumForFile(
        `${SHA256_A}  uv-aarch64-apple-darwin.tar.gz\n`,
        "uv-aarch64-apple-darwin.tar.gz",
      ),
    ).toBe(SHA256_A);

    expect(
      checksumForFile(
        [
          `${SHA256_A}  node-v22.12.0-darwin-x64.tar.gz`,
          `${SHA256_B} *node-v22.12.0-darwin-arm64.tar.gz`,
        ].join("\n"),
        "node-v22.12.0-darwin-arm64.tar.gz",
      ),
    ).toBe(SHA256_B);
  });

  it.each([
    {
      name: "missing filename",
      contents: `${SHA256_A}  other.tar.gz\n`,
      error: /no SHA-256 checksum found/i,
    },
    {
      name: "malformed digest",
      contents: "not-a-sha256  archive.tar.gz\n",
      error: /no SHA-256 checksum found/i,
    },
    {
      name: "duplicate entries",
      contents: `${SHA256_A}  archive.tar.gz\n${SHA256_B}  archive.tar.gz\n`,
      error: /multiple SHA-256 checksums found/i,
    },
  ])("rejects $name", ({ contents, error }) => {
    expect(() => checksumForFile(contents, "archive.tar.gz")).toThrow(error);
  });

  it("accepts the expected archive and rejects changed bytes", async () => {
    tempDir = await mkdtemp(join(tmpdir(), "download-integrity-"));
    const archive = join(tempDir, "archive.tar.gz");
    await writeFile(archive, "trusted archive");

    await expect(
      verifyFileSha256(archive, TRUSTED_ARCHIVE_SHA256),
    ).resolves.toBeUndefined();
    await expect(verifyFileSha256(archive, SHA256_A)).rejects.toThrow(
      /SHA-256 mismatch/,
    );
  });
});
