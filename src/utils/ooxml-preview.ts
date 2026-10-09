/**
 * Reader for OOXML documents (`.docx` / `.xlsx` / `.pptx`).
 *
 * These are ZIP containers of XML parts, so a preview needs a ZIP reader and an
 * XML reader. Rather than pull in a renderer for each format — `xlsx` (SheetJS)
 * carries ReDoS/prototype-pollution advisories and all of them are large — this
 * uses the platform's own `DecompressionStream("deflate-raw")` and `DOMParser`.
 * That keeps the feature to zero new dependencies and one lazy chunk.
 *
 * The output is a plain structured outline (headings, paragraphs, sheets, rows,
 * slides) that the card renders as styled text. It is deliberately not a
 * pixel-faithful reconstruction: the goal is to let a reviewer see what the
 * document says without leaving the conversation.
 */

export type OoxmlKind = "docx" | "xlsx" | "pptx";

export type OoxmlBlock =
  | { type: "heading"; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list-item"; text: string; ordered: boolean };

export type OoxmlSlide = {
  index: number;
  title: string | null;
  lines: string[];
};

export type OoxmlPreview = {
  kind: OoxmlKind;
  /** Word: the document body in order. Empty for the other kinds. */
  blocks: OoxmlBlock[];
  /** Excel: one entry per worksheet. */
  sheets: { name: string; rows: string[][] }[];
  /** PowerPoint: one entry per slide. */
  slides: OoxmlSlide[];
  /** True when a limit below truncated the output. */
  truncated: boolean;
};

// Caps so a pathological document cannot stall the render. Word keeps the most
// blocks because prose is the whole point; sheets and slides are capped tighter.
const MAX_BLOCKS = 400;
const MAX_SHEETS = 20;
const MAX_ROWS_PER_SHEET = 200;
const MAX_COLUMNS_PER_ROW = 40;
const MAX_SLIDES = 60;
const MAX_LINES_PER_SLIDE = 40;

const XML_MIME = "application/xml";

/**
 * Cap on the decompressed size of any single ZIP part. A ZIP's declared
 * uncompressed size is attacker-controlled, so a small compressed entry can
 * expand without bound ("zip bomb"); `DecompressionStream` would happily buffer
 * the whole result before the render caps below ever apply. Every part this
 * reader wants is XML text, so 8 MiB is generous for a real document and far
 * below what would threaten the tab.
 *
 * Applied to stored entries too: those skip inflation, but their declared
 * compressed size is still attacker-controlled and a hand-crafted header can
 * claim far more than the archive holds.
 */
export const MAX_INFLATED_PART_BYTES = 8 * 1024 * 1024;

/**
 * Cap on the *compressed* size of an Office document this preview will accept.
 * The container is downloaded whole before it is unpacked, so without this an
 * arbitrarily large `.docx` from the workspace fileserver would be buffered into
 * memory in one shot. A real OOXML package is a few MiB at most.
 */
export const MAX_OOXML_DOWNLOAD_BYTES = 32 * 1024 * 1024;

/**
 * Read a `fetch` response into an `ArrayBuffer`, refusing anything larger than
 * `maxBytes`. Prefers the declared `Content-Length` (a cheap early reject), then
 * streams the body so the ceiling is enforced while bytes arrive — a server that
 * omits or under-reports the length cannot make us buffer an unbounded body.
 */
export async function readBoundedArrayBuffer(
  response: Response,
  maxBytes: number,
): Promise<ArrayBuffer> {
  const declared = Number(response.headers?.get?.("content-length") ?? "");
  if (Number.isFinite(declared) && declared > maxBytes) {
    // Rejecting on the declared length happens before we ever read the body, so
    // the response stream is still open. Cancel it (best effort) or the browser
    // keeps downloading a file this preview will never parse.
    try {
      await response.body?.cancel();
    } catch {
      // The size error is the actionable outcome; a failed cancel is not.
    }
    throw new Error(`File exceeds ${maxBytes} bytes`);
  }

  const body = response.body;
  if (!body || typeof body.getReader !== "function") {
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength > maxBytes) {
      throw new Error(`File exceeds ${maxBytes} bytes`);
    }
    return buffer;
  }

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.length) continue;
      total += value.length;
      if (total > maxBytes) {
        await reader.cancel();
        throw new Error(`File exceeds ${maxBytes} bytes`);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

interface ZipEntry {
  name: string;
  /** Bytes of the entry's decompressed content. */
  bytes: () => Promise<Uint8Array>;
}

/**
 * Minimal ZIP central-directory reader. Only the entries we ask for are
 * inflated, and only stored (method 0) and deflated (method 8) entries are
 * understood — which is every part inside an OOXML package.
 */
function readZipEntries(buffer: ArrayBuffer): Map<string, ZipEntry> {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Locate the end-of-central-directory record by scanning backwards; its
  // signature is unique enough that a 64 KiB window is ample.
  const EOCD_SIGNATURE = 0x06054b50;
  const scanFrom = Math.max(0, buffer.byteLength - 0x10000 - 22);
  let eocd = -1;
  for (let i = buffer.byteLength - 22; i >= scanFrom; i -= 1) {
    if (view.getUint32(i, true) === EOCD_SIGNATURE) {
      eocd = i;
      break;
    }
  }
  if (eocd === -1)
    throw new Error("Not a ZIP archive (no end-of-central-directory)");

  const entryCount = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  const entries = new Map<string, ZipEntry>();

  for (let i = 0; i < entryCount; i += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) break;
    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localOffset = view.getUint32(offset + 42, true);
    const name = new TextDecoder().decode(
      bytes.subarray(offset + 46, offset + 46 + nameLength),
    );

    entries.set(name, {
      name,
      bytes: async () => {
        // The local header repeats the name/extra lengths, which can differ
        // from the central directory's, so read them from the local header.
        const localNameLength = view.getUint16(localOffset + 26, true);
        const localExtraLength = view.getUint16(localOffset + 28, true);
        const dataStart = localOffset + 30 + localNameLength + localExtraLength;
        const compressed = bytes.subarray(
          dataStart,
          dataStart + compressedSize,
        );
        if (method === 0) {
          // A stored part is returned as-is, so it bypasses `inflateRaw`'s
          // running cap. Enforce the same ceiling here, or an oversized stored
          // XML part flows straight into `DOMParser`.
          if (compressed.length > MAX_INFLATED_PART_BYTES) {
            throw new Error(
              `ZIP entry expands beyond ${MAX_INFLATED_PART_BYTES} bytes`,
            );
          }
          return compressed;
        }
        if (method !== 8) throw new Error(`Unsupported ZIP method ${method}`);
        return inflateRaw(compressed);
      },
    });

    offset += 46 + nameLength + extraLength + commentLength;
  }

  return entries;
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

async function inflateRaw(compressed: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("DecompressionStream is unavailable");
  }
  const stream = singleChunkStream(compressed).pipeThrough(
    // lib.dom types the transformer's writable side as `BufferSource`, which
    // TS will not unify with our `Uint8Array` chunks; the runtime contract is
    // identical.
    new DecompressionStream("deflate-raw") as unknown as ReadableWritablePair<
      Uint8Array,
      Uint8Array
    >,
  );

  // Stream the output so the cap is enforced *while* inflating, rather than
  // after buffering the whole part. `Response.arrayBuffer()` would materialize
  // an unbounded result before we could reject it.
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.length) continue;
      total += value.length;
      if (total > MAX_INFLATED_PART_BYTES) {
        await reader.cancel();
        throw new Error(
          `ZIP entry expands beyond ${MAX_INFLATED_PART_BYTES} bytes`,
        );
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function parseXml(text: string): Document {
  return new DOMParser().parseFromString(text, XML_MIME);
}

/**
 * All text runs under `root`, joined with the paragraph and line breaks that
 * the OOXML formats express as sibling elements rather than characters. Without
 * this, words in adjacent runs would run together, and PowerPoint paragraphs
 * (`<a:p>`) would collapse onto one line.
 */
function textOfRuns(root: Element): string {
  let out = "";
  for (const node of Array.from(root.getElementsByTagName("*"))) {
    switch (node.localName) {
      case "t":
        out += node.textContent ?? "";
        break;
      case "p":
        // A paragraph boundary. `getElementsByTagName("*")` only walks
        // descendants, so the element this was called on is not itself seen.
        out += "\n";
        break;
      case "br":
      case "cr":
        out += "\n";
        break;
      case "tab":
        out += "\t";
        break;
      default:
        break;
    }
  }
  return out;
}

function documentPart(entries: Map<string, ZipEntry>): Promise<Uint8Array> {
  const entry =
    entries.get("word/document.xml") ?? entries.get("word/document2.xml");
  if (!entry) throw new Error("word/document.xml is missing");
  return entry.bytes();
}

function blockFromParagraph(paragraph: Element): OoxmlBlock | null {
  const text = textOfRuns(paragraph).trim();
  if (!text) return null;

  const style = paragraph
    .getElementsByTagName("w:pStyle")[0]
    ?.getAttribute("w:val");
  const isHeading = Boolean(
    style && /^Heading|^heading|^Ttulo|^berschrift/.test(style),
  );
  if (isHeading) return { type: "heading", text };

  const numbering = paragraph.getElementsByTagName("w:numPr")[0];
  if (numbering) {
    const ordered = Boolean(
      numbering.getElementsByTagName("w:numId")[0]?.getAttribute("w:val"),
    );
    return { type: "list-item", text, ordered };
  }

  return { type: "paragraph", text };
}

async function readDocx(entries: Map<string, ZipEntry>): Promise<OoxmlPreview> {
  const xml = parseXml(new TextDecoder().decode(await documentPart(entries)));
  const body = xml.getElementsByTagName("w:body")[0];
  const blocks: OoxmlBlock[] = [];
  let truncated = false;

  if (body) {
    for (const child of Array.from(body.children)) {
      if (child.localName === "p") {
        const block = blockFromParagraph(child);
        if (block) blocks.push(block);
      } else if (child.localName === "tbl") {
        // Tables become one paragraph per row so the cell text stays visible
        // without pretending to reconstruct the grid. A single table can hold
        // thousands of rows, so the cap is checked inside the row loop too —
        // otherwise one table emits the whole document into the chat.
        for (const row of Array.from(child.getElementsByTagName("w:tr"))) {
          if (blocks.length >= MAX_BLOCKS) break;
          const cells = Array.from(row.getElementsByTagName("w:tc"))
            .map((cell) => textOfRuns(cell).trim())
            .filter(Boolean);
          if (cells.length)
            blocks.push({ type: "paragraph", text: cells.join(" · ") });
        }
      }
      if (blocks.length >= MAX_BLOCKS) {
        truncated = true;
        break;
      }
    }
  }

  return { kind: "docx", blocks, sheets: [], slides: [], truncated };
}

async function readSharedStrings(
  entries: Map<string, ZipEntry>,
): Promise<string[]> {
  const entry = entries.get("xl/sharedStrings.xml");
  if (!entry) return [];
  const xml = parseXml(new TextDecoder().decode(await entry.bytes()));
  return Array.from(xml.getElementsByTagName("si")).map((si) => textOfRuns(si));
}

/** Excel stores numbers and dates as raw values; only `t="s"` is a string index. */
function cellValue(cell: Element, sharedStrings: string[]): string {
  const type = cell.getAttribute("t");
  if (type === "inlineStr") return textOfRuns(cell).trim();
  const value = cell.getElementsByTagName("v")[0]?.textContent ?? "";
  if (type === "s") {
    const index = Number.parseInt(value, 10);
    return sharedStrings[index] ?? "";
  }
  return value;
}

/**
 * Zero-based column index from a cell reference like `A1`, `C7`, `AA3`.
 * Excel omits empty cells from the XML, so each present cell's `r` attribute
 * is the only thing that says which column it belongs to — without it, values
 * after a gap would shift left under the wrong heading.
 */
function columnIndexFromRef(ref: string | null): number | null {
  const match = /^([A-Za-z]+)/.exec(ref ?? "");
  if (!match) return null;
  let index = 0;
  for (const char of match[1].toUpperCase()) {
    index = index * 26 + (char.charCodeAt(0) - 64);
  }
  return index - 1;
}

/**
 * Resolve each workbook sheet's part path through its `r:id` relationship.
 * Worksheet filenames do not encode workbook order — the relationship target
 * does — so pairing `sheetN.xml` by loop index can show the wrong sheet's rows.
 * Targets are resolved relative to the workbook part (`xl/workbook.xml`), so
 * `worksheets/sheet2.xml` and an absolute `/xl/worksheets/sheet2.xml` both map
 * to the same entry.
 */
async function resolveWorkbookSheetPaths(
  entries: Map<string, ZipEntry>,
  workbook: Document,
): Promise<Map<string, string>> {
  const targets = new Map<string, string>();
  const relsEntry = entries.get("xl/_rels/workbook.xml.rels");
  if (relsEntry) {
    const rels = parseXml(new TextDecoder().decode(await relsEntry.bytes()));
    for (const rel of Array.from(rels.getElementsByTagName("Relationship"))) {
      targets.set(
        rel.getAttribute("Id") ?? "",
        rel.getAttribute("Target") ?? "",
      );
    }
  }

  const paths = new Map<string, string>();
  for (const sheet of Array.from(workbook.getElementsByTagName("sheet"))) {
    const rid = sheet.getAttribute("r:id") ?? "";
    const target = targets.get(rid);
    if (!target) continue;
    const normalized = normalizeWorkbookTarget(target);
    if (normalized) paths.set(rid, normalized);
  }
  return paths;
}

/** Normalize a relationship target to its ZIP entry name, relative to `xl/`. */
function normalizeWorkbookTarget(target: string): string | null {
  const cleaned = target.replace(/^\.\//, "");
  const absolute = cleaned.startsWith("/") ? cleaned.slice(1) : `xl/${cleaned}`;
  const segments: string[] = [];
  for (const segment of absolute.split("/")) {
    if (segment === "." || segment === "") continue;
    if (segment === "..") segments.pop();
    else segments.push(segment);
  }
  return segments.join("/");
}

async function readXlsx(entries: Map<string, ZipEntry>): Promise<OoxmlPreview> {
  const workbookEntry = entries.get("xl/workbook.xml");
  if (!workbookEntry) throw new Error("xl/workbook.xml is missing");
  const workbook = parseXml(
    new TextDecoder().decode(await workbookEntry.bytes()),
  );
  const sharedStrings = await readSharedStrings(entries);
  const sheetPaths = await resolveWorkbookSheetPaths(entries, workbook);

  const sheetElements = Array.from(workbook.getElementsByTagName("sheet"));
  const sheets: OoxmlPreview["sheets"] = [];
  let truncated = false;

  for (const [index, sheet] of sheetElements.entries()) {
    if (sheets.length >= MAX_SHEETS) {
      truncated = true;
      break;
    }
    const name = sheet.getAttribute("name") ?? `Sheet ${index + 1}`;
    // Prefer the relationship target; fall back to the legacy `sheetN.xml`
    // guess only when the workbook carries no relationship for this sheet.
    const rid = sheet.getAttribute("r:id") ?? "";
    const part =
      (sheetPaths.has(rid) ? entries.get(sheetPaths.get(rid)!) : undefined) ??
      entries.get(`xl/worksheets/sheet${index + 1}.xml`);
    if (!part) {
      sheets.push({ name, rows: [] });
      continue;
    }
    const xml = parseXml(new TextDecoder().decode(await part.bytes()));
    const rows: string[][] = [];
    for (const row of Array.from(xml.getElementsByTagName("row"))) {
      if (rows.length >= MAX_ROWS_PER_SHEET) {
        truncated = true;
        break;
      }
      // Place each cell at its real column, filling gaps with empty strings,
      // so a sparse row still lines up under the right headings.
      const cells: string[] = [];
      for (const cell of Array.from(row.getElementsByTagName("c"))) {
        const column = columnIndexFromRef(cell.getAttribute("r"));
        if (column === null) {
          cells.push(cellValue(cell, sharedStrings));
          continue;
        }
        if (column >= MAX_COLUMNS_PER_ROW) continue;
        while (cells.length < column) cells.push("");
        cells[column] = cellValue(cell, sharedStrings);
      }
      // Keep a row only if it has something in it; blank spacer rows are noise.
      if (cells.some((value) => value !== "")) rows.push(cells);
    }
    sheets.push({ name, rows });
  }

  return { kind: "xlsx", blocks: [], sheets, slides: [], truncated };
}

async function readPptx(entries: Map<string, ZipEntry>): Promise<OoxmlPreview> {
  // Slide order is defined by the presentation's sldIdLst, not by filename
  // order, so read the mapping and fall back to numeric order if it is absent.
  const presentationEntry = entries.get("ppt/presentation.xml");
  let order: number[] = [];
  if (presentationEntry) {
    const presentation = parseXml(
      new TextDecoder().decode(await presentationEntry.bytes()),
    );
    const relsEntry = entries.get("ppt/_rels/presentation.xml.rels");
    const relTargets = new Map<string, string>();
    if (relsEntry) {
      const rels = parseXml(new TextDecoder().decode(await relsEntry.bytes()));
      for (const rel of Array.from(rels.getElementsByTagName("Relationship"))) {
        relTargets.set(
          rel.getAttribute("Id") ?? "",
          rel.getAttribute("Target") ?? "",
        );
      }
    }
    order = Array.from(presentation.getElementsByTagName("p:sldId"))
      .map((sldId) => {
        const rid = sldId.getAttribute("r:id") ?? "";
        const target = relTargets.get(rid) ?? "";
        const match = /slide(\d+)\.xml$/.exec(target);
        return match ? Number.parseInt(match[1], 10) : 0;
      })
      .filter((n) => n > 0);
  }
  if (!order.length) {
    order = Array.from(entries.keys())
      .map((name) => /^ppt\/slides\/slide(\d+)\.xml$/.exec(name)?.[1])
      .filter((value): value is string => Boolean(value))
      .map((value) => Number.parseInt(value, 10))
      .sort((a, b) => a - b);
  }

  const slides: OoxmlSlide[] = [];
  let truncated = false;

  for (const [position, slideNumber] of order.entries()) {
    if (slides.length >= MAX_SLIDES) {
      truncated = true;
      break;
    }
    const part = entries.get(`ppt/slides/slide${slideNumber}.xml`);
    if (!part) continue;
    const xml = parseXml(new TextDecoder().decode(await part.bytes()));

    // The title placeholder is the slide's own headline; everything else is
    // body text. The placeholder type is declared on the shape's `p:ph`.
    let title: string | null = null;
    const lines: string[] = [];
    for (const shape of Array.from(xml.getElementsByTagName("p:sp"))) {
      const text = textOfRuns(shape).trim();
      if (!text) continue;
      const isTitle = /(^|\s)(title|ctrTitle)(\s|$)/.test(
        shape.getElementsByTagName("p:ph")[0]?.getAttribute("type") ?? "",
      );
      if (isTitle && !title) {
        title = text;
        continue;
      }
      for (const line of text.split("\n")) {
        if (line.trim()) lines.push(line.trim());
        if (lines.length >= MAX_LINES_PER_SLIDE) break;
      }
      if (lines.length >= MAX_LINES_PER_SLIDE) break;
    }

    slides.push({ index: position + 1, title, lines });
  }

  return { kind: "pptx", blocks: [], sheets: [], slides, truncated };
}

export async function readOoxmlPreview(
  kind: OoxmlKind,
  buffer: ArrayBuffer,
): Promise<OoxmlPreview> {
  const entries = readZipEntries(buffer);
  switch (kind) {
    case "docx":
      return readDocx(entries);
    case "xlsx":
      return readXlsx(entries);
    case "pptx":
      return readPptx(entries);
  }
}
