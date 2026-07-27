import { describe, expect, it } from "vitest";
import {
  getClipboardFiles,
  isPastedClipboardImage,
  normalizePastedFile,
  partitionImagesForUpload,
} from "#/components/features/chat/utils/chat-input.utils";

function createFileList(files: File[]): FileList {
  const list = {
    length: files.length,
    item: (index: number) => files[index] ?? null,
  } as FileList & Record<number, File>;
  for (let i = 0; i < files.length; i += 1) {
    list[i] = files[i];
  }
  return list;
}

function createMockDataTransfer({
  files = [],
  items = [],
}: {
  files?: File[];
  items?: Array<{ kind: string; type: string; file: File | null }>;
}): DataTransfer {
  const fileItems = items.map((entry) => ({
    kind: entry.kind,
    type: entry.type,
    getAsFile: () => entry.file,
  }));

  return {
    files: createFileList(files),
    items: fileItems as unknown as DataTransferItemList,
    getData: () => "",
  } as unknown as DataTransfer;
}

describe("normalizePastedFile", () => {
  it("returns the file unchanged when it already has a name", () => {
    const file = new File(["x"], "photo.png", { type: "image/png" });
    expect(normalizePastedFile(file)).toBe(file);
  });

  it("assigns a generated name for unnamed clipboard images", () => {
    const file = new File(["x"], "", { type: "image/png" });
    const normalized = normalizePastedFile(file);
    expect(normalized.name).toMatch(/^pasted-image-\d+\.png$/);
    expect(normalized.type).toBe("image/png");
  });
});

describe("isPastedClipboardImage", () => {
  it("returns true for normalized clipboard screenshot names", () => {
    const file = new File(["x"], "pasted-image-1710000000000.png", {
      type: "image/png",
    });
    expect(isPastedClipboardImage(file)).toBe(true);
  });

  it("returns false for images picked from the file dialog", () => {
    const file = new File(["x"], "photo.png", { type: "image/png" });
    expect(isPastedClipboardImage(file)).toBe(false);
  });
});

describe("partitionImagesForUpload", () => {
  it("splits marked images into the file-upload bucket", () => {
    const embed = new File(["a"], "embed.png", { type: "image/png" });
    const upload = new File(["b"], "upload.png", { type: "image/png" });

    const result = partitionImagesForUpload([embed, upload], ["upload.png"]);

    expect(result.imagesToEmbed).toEqual([embed]);
    expect(result.imagesAsFiles).toEqual([upload]);
  });
});

describe("getClipboardFiles", () => {
  it("reads from clipboardData.files when present", () => {
    const file = new File(["x"], "doc.txt", { type: "text/plain" });
    const clipboard = createMockDataTransfer({ files: [file] });

    expect(getClipboardFiles(clipboard)).toEqual([file]);
  });

  it("falls back to clipboard items for screenshot-style image paste", () => {
    const image = new File(["pixels"], "", { type: "image/png" });
    const clipboard = createMockDataTransfer({
      items: [{ kind: "file", type: "image/png", file: image }],
    });

    const result = getClipboardFiles(clipboard);
    expect(result).toHaveLength(1);
    expect(result[0].type).toBe("image/png");
    expect(result[0].name).toMatch(/^pasted-image-\d+\.png$/);
  });

  it("ignores non-file clipboard items", () => {
    const clipboard = createMockDataTransfer({
      items: [{ kind: "string", type: "text/plain", file: null }],
    });

    expect(getClipboardFiles(clipboard)).toEqual([]);
  });
});
