/**
 * Packing a handful of text files into a `.tar.gz`, in the browser.
 *
 * The automation service accepts a gzipped tar and nothing else, and the only
 * archives built here are the few small files a catalog bundle ships, so this
 * writes the original POSIX ustar format directly rather than pulling in a tar
 * library: 512-byte header, name and metadata as space-padded octal, content
 * padded to the next 512-byte boundary, two zero blocks to end the archive.
 * Gzip is the platform's own `CompressionStream`.
 *
 * Deliberately not general-purpose. Names must fit ustar's 100-byte field, and
 * only regular files are written - no directories, links, or long-name
 * extensions - because a bundle that needed any of those would be packed by
 * something that also has to unpack them.
 */

const BLOCK_SIZE = 512;
const NAME_FIELD_SIZE = 100;
const CHECKSUM_OFFSET = 148;
const CHECKSUM_SIZE = 8;

export interface TarFile {
  /** Path inside the archive. Must fit ustar's 100-byte name field. */
  name: string;
  content: string;
  /** Defaults to 0o644. A setup script wants 0o755. */
  mode?: number;
}

function header(file: TarFile, size: number): Uint8Array {
  const block = new Uint8Array(BLOCK_SIZE);

  const ascii = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index += 1) {
      block[offset + index] = value.charCodeAt(index) & 0x7f;
    }
  };
  /** ustar writes numbers as octal, NUL-terminated, right-aligned with zeros. */
  const octal = (offset: number, size_: number, value: number): void =>
    ascii(offset, value.toString(8).padStart(size_ - 1, "0"));

  ascii(0, file.name);
  octal(100, 8, file.mode ?? 0o644);
  octal(108, 8, 0); // uid
  octal(116, 8, 0); // gid
  octal(124, 12, size);
  // A fixed mtime keeps the archive byte-identical for identical inputs, so a
  // re-upload of an unchanged bundle is visibly unchanged.
  octal(136, 12, 0);
  block[156] = "0".charCodeAt(0); // regular file
  ascii(257, "ustar");
  ascii(263, "00");

  // The checksum is computed with its own field read as spaces, then written
  // into it as octal followed by NUL and a space.
  block.fill(0x20, CHECKSUM_OFFSET, CHECKSUM_OFFSET + CHECKSUM_SIZE);
  const sum = block.reduce((total, byte) => total + byte, 0);
  ascii(CHECKSUM_OFFSET, sum.toString(8).padStart(6, "0"));
  block[CHECKSUM_OFFSET + 6] = 0;
  block[CHECKSUM_OFFSET + 7] = 0x20;

  return block;
}

/** The uncompressed archive. Exported for tests; callers want `packTarGzip`. */
export function packTar(files: readonly TarFile[]): Uint8Array<ArrayBuffer> {
  const encoder = new TextEncoder();
  const blocks: Uint8Array[] = [];

  for (const file of files) {
    if (encoder.encode(file.name).length > NAME_FIELD_SIZE) {
      throw new Error(`tar: name too long for a ustar header: ${file.name}`);
    }
    const content = encoder.encode(file.content);
    blocks.push(header(file, content.length));
    const padded = new Uint8Array(
      Math.ceil(content.length / BLOCK_SIZE) * BLOCK_SIZE,
    );
    padded.set(content);
    blocks.push(padded);
  }

  // Two zero blocks mark the end of the archive.
  blocks.push(new Uint8Array(BLOCK_SIZE * 2));

  const total = blocks.reduce((size, block) => size + block.length, 0);
  const archive = new Uint8Array(new ArrayBuffer(total));
  let offset = 0;
  for (const block of blocks) {
    archive.set(block, offset);
    offset += block.length;
  }
  return archive;
}

/** The gzipped archive, ready to POST as `application/gzip`. */
export async function packTarGzip(
  files: readonly TarFile[],
): Promise<Uint8Array> {
  const tar = packTar(files);
  // Streamed from the bytes rather than through a Blob: a Blob's stream() is
  // absent in the jsdom test environment, and the archive is one chunk anyway.
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(tar);
      controller.close();
    },
  });
  const compressed = new Response(
    source.pipeThrough(new CompressionStream("gzip")),
  );
  return new Uint8Array(await compressed.arrayBuffer());
}
