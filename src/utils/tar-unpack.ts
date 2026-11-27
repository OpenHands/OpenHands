/**
 * Reading a script bundle back out of the `.tar.gz` the automation service
 * stores, in the browser.
 *
 * The counterpart of `tar-gzip.ts`, and as deliberately narrow: the archives
 * read here are the few small files an automation ships, so this walks POSIX
 * ustar headers directly rather than pulling in a tar library. Gunzip is the
 * platform's own `DecompressionStream`.
 *
 * Only regular files are returned. Directories, links, and the pax / GNU
 * extension headers that carry long names are skipped without being applied,
 * so a member whose path exceeds ustar's 155 + 100 bytes surfaces under its
 * truncated header name — acceptable for a bundle, where names are short.
 */

const BLOCK_SIZE = 512;
const USTAR_MAGIC = "ustar";
const GZIP_MAGIC = [0x1f, 0x8b] as const;

/** Header fields at their ustar offsets. */
const NAME = { offset: 0, size: 100 };
const SIZE = { offset: 124, size: 12 };
const TYPEFLAG_OFFSET = 156;
const MAGIC = { offset: 257, size: 5 };
const PREFIX = { offset: 345, size: 155 };

const REGULAR_FILE_TYPEFLAGS = new Set(["0", "\0"]);

export interface TarEntry {
  /** Path inside the archive, without a leading `./`. */
  path: string;
  /** Size in bytes of the member's content. */
  size: number;
  /** The content decoded as UTF-8, or null when it is not valid UTF-8. */
  text: string | null;
}

/** The bytes are not a tar archive (nor a gzip of one). */
export class TarFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TarFormatError";
  }
}

/** The unpacked archive exceeds the caller's budget. */
export class TarTooLargeError extends Error {
  constructor(maxTotalBytes: number) {
    super(`tar: archive exceeds ${maxTotalBytes} bytes unpacked`);
    this.name = "TarTooLargeError";
  }
}

const asciiDecoder = new TextDecoder("ascii");
const utf8Decoder = new TextDecoder("utf-8", { fatal: true });

function isGzip(bytes: Uint8Array): boolean {
  return bytes[0] === GZIP_MAGIC[0] && bytes[1] === GZIP_MAGIC[1];
}

async function gunzip(
  bytes: Uint8Array<ArrayBuffer>,
): Promise<Uint8Array<ArrayBuffer>> {
  // Streamed from the bytes rather than through a Blob, mirroring
  // `packTarGzip`: a Blob's stream() is absent in the jsdom test environment.
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const inflated = new Response(
    source.pipeThrough(new DecompressionStream("gzip")),
  );
  return new Uint8Array(await inflated.arrayBuffer());
}

/** A NUL-terminated header string field. */
function stringField(
  header: Uint8Array,
  field: { offset: number; size: number },
): string {
  const raw = asciiDecoder.decode(
    header.subarray(field.offset, field.offset + field.size),
  );
  const end = raw.indexOf("\0");
  return end === -1 ? raw : raw.slice(0, end);
}

/** The member's path as ustar spells it: `prefix/name` when a prefix is set. */
function entryPath(header: Uint8Array): string {
  const name = stringField(header, NAME);
  const prefix = stringField(header, PREFIX);
  const joined = prefix ? `${prefix}/${name}` : name;
  return joined.replace(/^\.\//, "");
}

function entrySize(header: Uint8Array): number {
  // ustar writes sizes as octal; the base-256 extension (high bit set) is
  // for members far larger than any bundle, so it is refused rather than
  // parsed.
  if (header[SIZE.offset] & 0x80) {
    throw new TarFormatError("tar: base-256 size field is not supported");
  }
  const size = parseInt(stringField(header, SIZE).trim() || "0", 8);
  if (!Number.isFinite(size) || size < 0) {
    throw new TarFormatError("tar: malformed size field");
  }
  return size;
}

function decodeText(bytes: Uint8Array): string | null {
  try {
    return utf8Decoder.decode(bytes);
  } catch {
    return null;
  }
}

/**
 * Unpack a gzipped (or plain) ustar archive into its regular files.
 *
 * Throws `TarFormatError` when the bytes are not an archive — a response
 * that is really an error page, say — and `TarTooLargeError` once the
 * unpacked members exceed `maxTotalBytes`.
 */
export async function unpackTarGzip(
  bytes: Uint8Array<ArrayBuffer>,
  options: { maxTotalBytes: number },
): Promise<TarEntry[]> {
  const archive = isGzip(bytes) ? await gunzip(bytes) : bytes;
  const entries: TarEntry[] = [];
  let unpackedBytes = 0;

  let offset = 0;
  while (offset + BLOCK_SIZE <= archive.length) {
    const header = archive.subarray(offset, offset + BLOCK_SIZE);
    // Two zero blocks end the archive; one is enough to stop reading.
    if (header.every((byte) => byte === 0)) break;
    if (stringField(header, MAGIC) !== USTAR_MAGIC) {
      throw new TarFormatError("tar: not a ustar archive");
    }

    const size = entrySize(header);
    const contentStart = offset + BLOCK_SIZE;
    const contentEnd = contentStart + size;
    if (contentEnd > archive.length) {
      throw new TarFormatError("tar: member runs past the end of the archive");
    }

    const typeflag = String.fromCharCode(header[TYPEFLAG_OFFSET]);
    if (REGULAR_FILE_TYPEFLAGS.has(typeflag)) {
      unpackedBytes += size;
      if (unpackedBytes > options.maxTotalBytes) {
        throw new TarTooLargeError(options.maxTotalBytes);
      }
      entries.push({
        path: entryPath(header),
        size,
        text: decodeText(archive.subarray(contentStart, contentEnd)),
      });
    }

    offset = contentStart + Math.ceil(size / BLOCK_SIZE) * BLOCK_SIZE;
  }

  return entries;
}
