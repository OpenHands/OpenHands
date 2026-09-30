import { describe, expect, it } from "vitest";
import { makeStoredZip, makeZip } from "../helpers/make-ooxml-zip";
import {
  MAX_INFLATED_PART_BYTES,
  readOoxmlPreview,
} from "#/utils/ooxml-preview";

// jsdom has neither DecompressionStream nor DOMParser's XML support, so the
// reader's two platform primitives are stubbed from Node's own implementations.
// These are the real engines the browser uses, not mocks of our own logic.
if (typeof globalThis.DecompressionStream === "undefined") {
  const { DecompressionStream } = await import("node:stream/web");
  // @ts-expect-error assigning Node's implementation for the jsdom test env
  globalThis.DecompressionStream = DecompressionStream;
}

const W_NS =
  'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';

const DOCX_PARTS = {
  "[Content_Types].xml": `<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>`,
  "word/document.xml": `<?xml version="1.0"?><w:document ${W_NS}><w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Requirements</w:t></w:r></w:p>
    <w:p><w:r><w:t>Login must be </w:t></w:r><w:r><w:t>auditable</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>MFA</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t>Rate limiting</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Ana</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:p><w:r><w:t>Empty</w:t></w:r></w:p>
  </w:body></w:document>`,
};

const XLSX_PARTS = {
  "xl/workbook.xml": `<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Budget" sheetId="1" r:id="rId1"/><sheet name="Actuals" sheetId="2" r:id="rId2"/></sheets></workbook>`,
  "xl/sharedStrings.xml": `<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Item</t></si><si><t>Cost</t></si><si><t>Server</t></si></sst>`,
  "xl/worksheets/sheet1.xml": `<?xml version="1.0"?><worksheet><sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
    <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>1200</v></c></row>
    <row r="3"></row>
  </sheetData></worksheet>`,
  "xl/worksheets/sheet2.xml": `<?xml version="1.0"?><worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Done</t></is></c></row></sheetData></worksheet>`,
};

const PPTX_PARTS = {
  "ppt/presentation.xml": `<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId1"/></p:sldIdLst></p:presentation>`,
  "ppt/_rels/presentation.xml.rels": `<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>`,
  "ppt/slides/slide1.xml": `<?xml version="1.0"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>
    <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Overview</a:t></a:r></a:p></p:txBody></p:sp>
    <p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Goal</a:t></a:r></a:p><a:p><a:r><a:t>Scope</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld></p:sld>`,
  "ppt/slides/slide2.xml": `<?xml version="1.0"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>
    <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Plan</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld></p:sld>`,
};

describe("readOoxmlPreview", () => {
  it("outlines a Word document in order, splitting runs", async () => {
    const preview = await readOoxmlPreview("docx", await makeZip(DOCX_PARTS));

    expect(preview.kind).toBe("docx");
    expect(preview.blocks).toEqual([
      { type: "heading", text: "Requirements" },
      // Adjacent runs must not run together: "auditable" is a separate run.
      { type: "paragraph", text: "Login must be auditable" },
      { type: "list-item", text: "MFA", ordered: true },
      { type: "list-item", text: "Rate limiting", ordered: true },
      { type: "paragraph", text: "Owner · Ana" },
      { type: "paragraph", text: "Empty" },
    ]);
  });

  it("reads Excel sheets with shared and inline strings", async () => {
    const preview = await readOoxmlPreview("xlsx", await makeZip(XLSX_PARTS));

    expect(preview.sheets.map((sheet) => sheet.name)).toEqual([
      "Budget",
      "Actuals",
    ]);
    expect(preview.sheets[0].rows).toEqual([
      ["Item", "Cost"],
      ["Server", "1200"],
    ]);
    // A wholly empty row is dropped rather than rendered as a blank strip.
    expect(preview.sheets[1].rows).toEqual([["Done"]]);
  });

  it("orders PowerPoint slides by the presentation, not the filename", async () => {
    const preview = await readOoxmlPreview("pptx", await makeZip(PPTX_PARTS));

    // rId2 (slide2, "Plan") is listed first, so it must come out first.
    expect(preview.slides.map((slide) => slide.title)).toEqual([
      "Plan",
      "Overview",
    ]);
    expect(preview.slides[1].lines).toEqual(["Goal", "Scope"]);
  });

  it("places sparse Excel cells at their real columns", async () => {
    const parts = {
      ...XLSX_PARTS,
      "xl/worksheets/sheet1.xml": `<?xml version="1.0"?><worksheet><sheetData>
        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="s"><v>1</v></c></row>
        <row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>1200</v></c></row>
      </sheetData></worksheet>`,
    };
    const preview = await readOoxmlPreview("xlsx", await makeZip(parts));

    // `B` is absent from the XML, so it must stay an empty column rather than
    // letting "Cost" / "1200" slide left under the wrong heading.
    expect(preview.sheets[0].rows).toEqual([
      ["Item", "", "Cost"],
      ["Server", "", "1200"],
    ]);
  });

  it("resolves worksheet parts through rIds, not filenames", async () => {
    const parts = {
      ...XLSX_PARTS,
      "xl/workbook.xml": `<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Summary" sheetId="1" r:id="rId2"/><sheet name="Data" sheetId="2" r:id="rId1"/></sheets></workbook>`,
      "xl/_rels/workbook.xml.rels": `<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>`,
    };
    const preview = await readOoxmlPreview("xlsx", await makeZip(parts));

    // "Summary" is listed first but points at sheet2; index-pairing would have
    // shown it sheet1's rows.
    expect(preview.sheets.map((sheet) => sheet.name)).toEqual([
      "Summary",
      "Data",
    ]);
    expect(preview.sheets[0].rows).toEqual([["Done"]]);
    expect(preview.sheets[1].rows).toEqual([
      ["Item", "Cost"],
      ["Server", "1200"],
    ]);
  });

  it("rejects a ZIP part that inflates beyond the cap", async () => {
    // A zip bomb: highly compressible content whose *decompressed* size is far
    // past the cap, so the reader must refuse it instead of buffering it whole.
    const bomb = "A".repeat(9 * 1024 * 1024);
    const parts = {
      ...DOCX_PARTS,
      "word/document.xml": `<?xml version="1.0"?><w:document ${W_NS}><w:body><w:p><w:r><w:t>${bomb}</w:t></w:r></w:p></w:body></w:document>`,
    };

    await expect(
      readOoxmlPreview("docx", await makeZip(parts)),
    ).rejects.toThrow(/expands beyond/i);
  });

  it("caps a stored (uncompressed) part, which bypasses inflation", async () => {
    // `makeZip` always deflates, so build a stored entry by hand: a small
    // archive that *claims* a huge stored part must be refused, not handed to
    // `DOMParser` as an oversized string.
    const stored = makeStoredZip({
      "word/document.xml": "A".repeat(MAX_INFLATED_PART_BYTES + 1),
    });

    await expect(readOoxmlPreview("docx", stored)).rejects.toThrow(
      /expands beyond/i,
    );
  });

  it("rejects a buffer that is not a ZIP archive", async () => {
    await expect(
      readOoxmlPreview("docx", new TextEncoder().encode("not a zip").buffer),
    ).rejects.toThrow(/not a zip|end-of-central-directory/i);
  });

  it("stops at the block cap inside a single large table", async () => {
    // One table can hold thousands of rows, so the cap has to be enforced in
    // the row loop — checking it only after the table finishes would let a
    // single table emit the whole document into the chat.
    const rows = Array.from(
      { length: 1000 },
      (_, i) =>
        `<w:tr><w:tc><w:p><w:r><w:t>Row ${i}</w:t></w:r></w:p></w:tc></w:tr>`,
    ).join("");
    const bytes = await makeZip({
      "word/document.xml": `<?xml version="1.0"?><w:document ${W_NS}><w:body><w:tbl>${rows}</w:tbl></w:body></w:document>`,
    });

    const preview = await readOoxmlPreview("docx", bytes);

    expect(preview.blocks).toHaveLength(400);
    expect(preview.truncated).toBe(true);
  });
});
