import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { packTar, packTarGzip } from "#/utils/tar-gzip";

const BLOCK_SIZE = 512;
const decoder = new TextDecoder();

interface ParsedMember {
  name: string;
  mode: number;
  content: string;
  checksumMatches: boolean;
}

/**
 * Read an archive back the way tar does, so the test agrees with tar rather
 * than with the writer under test: fields at their ustar offsets, the checksum
 * recomputed with its own field read as spaces.
 */
function readTar(archive: Uint8Array): ParsedMember[] {
  const members: ParsedMember[] = [];
  const field = (block: Uint8Array, offset: number, size: number) =>
    decoder.decode(block.subarray(offset, offset + size)).replace(/\0.*$/, "");

  let offset = 0;
  while (offset + BLOCK_SIZE <= archive.length) {
    const header = archive.subarray(offset, offset + BLOCK_SIZE);
    if (header.every((byte) => byte === 0)) break;

    const size = parseInt(field(header, 124, 12).trim() || "0", 8);
    const recomputed = header.reduce(
      (total, byte, index) =>
        total + (index >= 148 && index < 156 ? 0x20 : byte),
      0,
    );

    members.push({
      name: field(header, 0, 100),
      mode: parseInt(field(header, 100, 8).trim() || "0", 8),
      content: decoder.decode(
        archive.subarray(offset + BLOCK_SIZE, offset + BLOCK_SIZE + size),
      ),
      checksumMatches:
        parseInt(field(header, 148, 8).trim() || "-1", 8) === recomputed,
    });
    offset += BLOCK_SIZE + Math.ceil(size / BLOCK_SIZE) * BLOCK_SIZE;
  }
  return members;
}

describe("packTar", () => {
  it("writes each file with its content, mode and a valid checksum", () => {
    // Act
    const archive = packTar([
      { name: "main.py", content: "print('hi')\n" },
      { name: "setup.sh", content: "#!/bin/bash\nset -e\n", mode: 0o755 },
    ]);

    // Assert
    expect(readTar(archive)).toEqual([
      {
        name: "main.py",
        mode: 0o644,
        content: "print('hi')\n",
        checksumMatches: true,
      },
      {
        name: "setup.sh",
        mode: 0o755,
        content: "#!/bin/bash\nset -e\n",
        checksumMatches: true,
      },
    ]);
  });

  it("pads content to the block size and ends with two zero blocks", () => {
    // Act
    const archive = packTar([{ name: "a.txt", content: "x" }]);

    // Assert: one header, one padded content block, two end-of-archive blocks.
    expect(archive.length).toBe(BLOCK_SIZE * 4);
    expect(archive.subarray(BLOCK_SIZE * 2).every((byte) => byte === 0)).toBe(
      true,
    );
  });

  it("survives multi-byte content, whose length is bytes and not characters", () => {
    // Arrange: three bytes in UTF-8, one character in JavaScript.
    const content = "héllo — ✓\n";

    // Act
    const members = readTar(packTar([{ name: "notes.md", content }]));

    // Assert
    expect(members[0].content).toBe(content);
  });

  it("writes a multi-byte name as the bytes a reader takes it back from", () => {
    // Arrange: the name is 8 bytes in UTF-8 and 7 characters in JavaScript,
    // and the length guard measures the bytes.
    const name = "café.py";

    // Act
    const members = readTar(packTar([{ name, content: "" }]));

    // Assert
    expect(members[0].name).toBe(name);
  });

  it("refuses a name that would not fit a ustar header", () => {
    // Arrange
    const name = `${"nested/".repeat(15)}main.py`;

    // Act + Assert
    expect(() => packTar([{ name, content: "" }])).toThrow(/name too long/);
  });

  it("is byte-identical for identical input, so an unchanged bundle re-uploads unchanged", () => {
    // Act
    const first = packTar([{ name: "main.py", content: "print(1)\n" }]);
    const second = packTar([{ name: "main.py", content: "print(1)\n" }]);

    // Assert
    expect(Buffer.from(first)).toEqual(Buffer.from(second));
  });
});

describe("packTarGzip", () => {
  it("produces gzip that decompresses to the same archive", async () => {
    // Arrange
    const files = [{ name: "main.py", content: "print('hi')\n" }];

    // Act
    const compressed = await packTarGzip(files);

    // Assert
    expect(compressed[0]).toBe(0x1f);
    expect(compressed[1]).toBe(0x8b);
    expect(readTar(new Uint8Array(gunzipSync(compressed)))).toEqual(
      readTar(packTar(files)),
    );
  });
});
