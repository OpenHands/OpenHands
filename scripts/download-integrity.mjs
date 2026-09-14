import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { basename } from "node:path";

const SHA256_PATTERN = /^[a-fA-F0-9]{64}$/;
const CHECKSUM_LINE_PATTERN = /^([a-fA-F0-9]{64})[ \t]+\*?(.+)$/;

export function checksumForFile(contents, filename) {
  const matches = [];

  for (const line of contents.split(/\r?\n/)) {
    const match = CHECKSUM_LINE_PATTERN.exec(line);
    if (match?.[2] === filename) {
      matches.push(match[1].toLowerCase());
    }
  }

  if (matches.length === 0) {
    throw new Error(`No SHA-256 checksum found for ${filename}`);
  }
  if (matches.length > 1) {
    throw new Error(`Multiple SHA-256 checksums found for ${filename}`);
  }

  return matches[0];
}

export async function verifyFileSha256(filePath, expectedSha256) {
  if (!SHA256_PATTERN.test(expectedSha256)) {
    throw new Error(`Invalid SHA-256 checksum for ${basename(filePath)}`);
  }

  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
  }

  const actualSha256 = hash.digest("hex");
  if (actualSha256 !== expectedSha256.toLowerCase()) {
    throw new Error(
      `SHA-256 mismatch for ${basename(filePath)}: expected ` +
        `${expectedSha256.toLowerCase()}, received ${actualSha256}`,
    );
  }
}
