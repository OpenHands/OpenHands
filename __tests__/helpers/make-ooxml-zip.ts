/**
 * Builds a real ZIP archive (deflate-raw entries) so tests can exercise the
 * OOXML reader against actual `.docx` / `.xlsx` / `.pptx` bytes rather than a
 * mock of the unpacking step. Mirrors the layout of an OOXML package: local
 * file headers, a central directory, and an end-of-central-directory record.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let value = i;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[i] = value >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** A single-chunk `ReadableStream`, so we never depend on `Blob.stream()`. */
function singleChunkStream(bytes: Uint8Array): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

async function deflateRaw(input: Uint8Array): Promise<Uint8Array> {
  const stream = singleChunkStream(input).pipeThrough(
    // See ooxml-preview.ts: lib.dom's BufferSource writable side is not
    // assignable to our Uint8Array chunks, though the runtime contract matches.
    new CompressionStream("deflate-raw") as unknown as ReadableWritablePair<
      Uint8Array,
      Uint8Array
    >,
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

interface Entry {
  name: string;
  data: Uint8Array;
  compressed: Uint8Array;
  crc: number;
  offset: number;
}

/** `parts` maps a ZIP entry name to its text content. */
export async function makeZip(
  parts: Record<string, string>,
): Promise<ArrayBuffer> {
  const encoder = new TextEncoder();
  const entries: Entry[] = [];
  const chunks: Uint8Array[] = [];
  let offset = 0;

  const push = (bytes: Uint8Array) => {
    chunks.push(bytes);
    offset += bytes.length;
  };

  for (const [name, content] of Object.entries(parts)) {
    const data = encoder.encode(content);
    const compressed = await deflateRaw(data);
    const nameBytes = encoder.encode(name);
    const entry: Entry = {
      name,
      data,
      compressed,
      crc: crc32(data),
      offset,
    };

    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true); // local file header signature
    header.setUint16(4, 20, true); // version needed
    header.setUint16(6, 0, true); // flags
    header.setUint16(8, 8, true); // method: deflate
    header.setUint16(10, 0, true); // mod time
    header.setUint16(12, 0, true); // mod date
    header.setUint32(14, entry.crc, true);
    header.setUint32(18, compressed.length, true);
    header.setUint32(22, data.length, true);
    header.setUint16(26, nameBytes.length, true);
    header.setUint16(28, 0, true); // extra length
    push(new Uint8Array(header.buffer));
    push(nameBytes);
    push(compressed);
    entries.push(entry);
  }

  const centralOffset = offset;
  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.name);
    const record = new DataView(new ArrayBuffer(46));
    record.setUint32(0, 0x02014b50, true); // central directory signature
    record.setUint16(4, 20, true); // version made by
    record.setUint16(6, 20, true); // version needed
    record.setUint16(8, 0, true); // flags
    record.setUint16(10, 8, true); // method
    record.setUint16(12, 0, true);
    record.setUint16(14, 0, true);
    record.setUint32(16, entry.crc, true);
    record.setUint32(20, entry.compressed.length, true);
    record.setUint32(24, entry.data.length, true);
    record.setUint16(28, nameBytes.length, true);
    record.setUint16(30, 0, true); // extra
    record.setUint16(32, 0, true); // comment
    record.setUint16(34, 0, true); // disk number
    record.setUint16(36, 0, true); // internal attrs
    record.setUint32(38, 0, true); // external attrs
    record.setUint32(42, entry.offset, true);
    push(new Uint8Array(record.buffer));
    push(nameBytes);
  }

  const centralSize = offset - centralOffset;
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(4, 0, true);
  eocd.setUint16(6, 0, true);
  eocd.setUint16(8, entries.length, true);
  eocd.setUint16(10, entries.length, true);
  eocd.setUint32(12, centralSize, true);
  eocd.setUint32(16, centralOffset, true);
  eocd.setUint16(20, 0, true);
  push(new Uint8Array(eocd.buffer));

  const total = new Uint8Array(offset);
  let cursor = 0;
  for (const chunk of chunks) {
    total.set(chunk, cursor);
    cursor += chunk.length;
  }
  return total.buffer;
}

/**
 * Builds a ZIP whose entries use the *stored* method (0), so the reader returns
 * the bytes without inflating them. Used to prove the per-part cap also applies
 * to stored entries, which bypass `inflateRaw`'s running check.
 */
export function makeStoredZip(parts: Record<string, string>): ArrayBuffer {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const entries: Entry[] = [];
  let offset = 0;

  const push = (bytes: Uint8Array) => {
    chunks.push(bytes);
    offset += bytes.length;
  };

  for (const [name, content] of Object.entries(parts)) {
    const data = encoder.encode(content);
    const nameBytes = encoder.encode(name);
    const entry: Entry = {
      name,
      data,
      compressed: data,
      crc: crc32(data),
      offset,
    };

    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true);
    header.setUint16(6, 0, true);
    header.setUint16(8, 0, true); // method 0: stored
    header.setUint32(14, entry.crc, true);
    header.setUint32(18, data.length, true);
    header.setUint32(22, data.length, true);
    header.setUint16(26, nameBytes.length, true);
    push(new Uint8Array(header.buffer));
    push(nameBytes);
    push(data);
    entries.push(entry);
  }

  const centralOffset = offset;
  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.name);
    const record = new DataView(new ArrayBuffer(46));
    record.setUint32(0, 0x02014b50, true);
    record.setUint16(4, 20, true);
    record.setUint16(6, 20, true);
    record.setUint16(8, 0, true);
    record.setUint16(10, 0, true); // method 0: stored
    record.setUint32(16, entry.crc, true);
    record.setUint32(20, entry.compressed.length, true);
    record.setUint32(24, entry.data.length, true);
    record.setUint16(28, nameBytes.length, true);
    record.setUint32(42, entry.offset, true);
    push(new Uint8Array(record.buffer));
    push(nameBytes);
  }

  const centralSize = offset - centralOffset;
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(8, entries.length, true);
  eocd.setUint16(10, entries.length, true);
  eocd.setUint32(12, centralSize, true);
  eocd.setUint32(16, centralOffset, true);
  push(new Uint8Array(eocd.buffer));

  const total = new Uint8Array(offset);
  let cursor = 0;
  for (const chunk of chunks) {
    total.set(chunk, cursor);
    cursor += chunk.length;
  }
  return total.buffer;
}
