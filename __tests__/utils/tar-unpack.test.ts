import { describe, expect, it } from "vitest";
import { packTar, packTarGzip } from "#/utils/tar-gzip";
import {
  TarFormatError,
  TarTooLargeError,
  unpackTarGzip,
} from "#/utils/tar-unpack";

const files = [
  { name: "main.py", content: "print('hi')\n" },
  { name: "lib/util.py", content: "x = 1\n" },
  { name: "setup.sh", content: "#!/bin/sh\n", mode: 0o755 },
];
const budget = { maxTotalBytes: 1024 * 1024 };

/** ustar header field offsets, as `packTar` writes them. */
const TYPEFLAG_OFFSET = 156;
const PREFIX_OFFSET = 345;

describe("unpackTarGzip", () => {
  it("reads the files back out of a gzipped bundle", async () => {
    // Arrange
    const archive = new Uint8Array(await packTarGzip(files));

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries).toEqual([
      { path: "main.py", size: 12, text: "print('hi')\n" },
      { path: "lib/util.py", size: 6, text: "x = 1\n" },
      { path: "setup.sh", size: 10, text: "#!/bin/sh\n" },
    ]);
  });

  it("reads an uncompressed tar as well", async () => {
    // Act
    const entries = await unpackTarGzip(packTar(files), budget);

    // Assert
    expect(entries.map((entry) => entry.path)).toEqual([
      "main.py",
      "lib/util.py",
      "setup.sh",
    ]);
  });

  it("strips a leading ./ from member names", async () => {
    // Act
    const entries = await unpackTarGzip(
      packTar([{ name: "./main.py", content: "1" }]),
      budget,
    );

    // Assert
    expect(entries[0].path).toBe("main.py");
  });

  it("joins the ustar prefix field onto the name", async () => {
    // Arrange: a header whose directory part lives in the prefix field.
    const archive = packTar([{ name: "util.py", content: "1" }]);
    archive.set(new TextEncoder().encode("lib"), PREFIX_OFFSET);

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("lib/util.py");
  });

  it("reads back a UTF-8 member name the packer wrote", async () => {
    // Arrange
    const archive = new Uint8Array(
      await packTarGzip([{ name: "café.py", content: "1" }]),
    );

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("café.py");
  });

  it("reads a UTF-8 prefix field", async () => {
    // Arrange
    const archive = packTar([{ name: "main.py", content: "1" }]);
    archive.set(new TextEncoder().encode("données"), PREFIX_OFFSET);

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("données/main.py");
  });

  it("reads a UTF-8 name up to its NUL, whatever follows", async () => {
    // Arrange: a byte that is not UTF-8 in the name field after the NUL.
    const archive = packTar([{ name: "café.py", content: "1" }]);
    archive[40] = 0xff;

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("café.py");
  });

  it("keeps a leading U+FEFF as part of a name", async () => {
    // Arrange
    const archive = packTar([{ name: "\uFEFFmain.py", content: "1" }]);

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("\uFEFFmain.py");
  });

  it("still reads a legacy Latin-1 member name", async () => {
    // Arrange: `café.py` with é as the single byte 0xe9, which is not UTF-8.
    const archive = packTar([{ name: "cafe.py", content: "1" }]);
    archive[3] = 0xe9;

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries[0].path).toBe("café.py");
  });

  it("skips members that are not regular files", async () => {
    // Arrange: mark the first member as a directory.
    const archive = packTar([
      { name: "lib", content: "" },
      { name: "main.py", content: "1" },
    ]);
    archive[TYPEFLAG_OFFSET] = "5".charCodeAt(0);

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries.map((entry) => entry.path)).toEqual(["main.py"]);
  });

  it("marks content that is not UTF-8 as binary", async () => {
    // Arrange: overwrite the member's bytes with an invalid sequence.
    const archive = packTar([{ name: "model.bin", content: "AAAA" }]);
    archive.fill(0xff, 512, 516);

    // Act
    const entries = await unpackTarGzip(archive, budget);

    // Assert
    expect(entries).toEqual([{ path: "model.bin", size: 4, text: null }]);
  });

  it("returns no entries for an empty archive", async () => {
    // Act
    const entries = await unpackTarGzip(packTar([]), budget);

    // Assert
    expect(entries).toEqual([]);
  });

  it("rejects bytes that are not an archive", async () => {
    // Arrange: an error page instead of a tarball.
    const html = new TextEncoder().encode("<html>".padEnd(1024, "x"));

    // Act + Assert
    await expect(
      unpackTarGzip(new Uint8Array(html), budget),
    ).rejects.toBeInstanceOf(TarFormatError);
  });

  it("rejects an archive that unpacks past the size budget", async () => {
    // Act + Assert
    await expect(
      unpackTarGzip(packTar(files), { maxTotalBytes: 20 }),
    ).rejects.toBeInstanceOf(TarTooLargeError);
  });
});
