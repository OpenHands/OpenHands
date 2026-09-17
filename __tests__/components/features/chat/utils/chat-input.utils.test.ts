import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearEmptyContent,
  clearFileInput,
  clearTextContent,
  ensureCursorVisible,
  focusContentEditableAtEnd,
  getClipboardFiles,
  getTextContent,
  isContentEmpty,
  isPastedClipboardImage,
  normalizePastedFile,
  partitionImagesForUpload,
} from "#/components/features/chat/utils/chat-input.utils";

afterEach(() => {
  vi.restoreAllMocks();
});

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
    expect(normalized.size).toBe(file.size);
  });

  it.each([
    ["image/jpeg", "jpg"],
    ["image/jpg", "jpg"],
    ["image/gif", "gif"],
    ["image/webp", "webp"],
    ["image/bmp", "bmp"],
    ["image/avif", "png"],
    ["application/octet-stream", "bin"],
  ])("uses the stable extension for %s", (type, extension) => {
    vi.spyOn(Date, "now").mockReturnValue(1234);
    const normalized = normalizePastedFile(
      new File(["x"], "   ", { type, lastModified: 99 }),
    );

    expect(normalized.name).toBe(`pasted-image-1234.${extension}`);
    expect(normalized.lastModified).toBe(99);
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

  it("matches normalized names case-insensitively", () => {
    expect(
      isPastedClipboardImage(
        new File(["x"], "PASTED-IMAGE-1710000000000.WEBP"),
      ),
    ).toBe(true);
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

  it("handles empty input and duplicate mark names", () => {
    expect(partitionImagesForUpload([], ["missing.png"])).toEqual({
      imagesToEmbed: [],
      imagesAsFiles: [],
    });
    const image = new File(["a"], "same.png", { type: "image/png" });
    expect(partitionImagesForUpload([image], ["same.png", "same.png"])).toEqual(
      {
        imagesToEmbed: [],
        imagesAsFiles: [image],
      },
    );
  });
});

describe("getClipboardFiles", () => {
  it("reads from clipboardData.files when present", () => {
    const file = new File(["x"], "doc.txt", { type: "text/plain" });
    const clipboard = createMockDataTransfer({ files: [file] });

    expect(getClipboardFiles(clipboard)).toEqual([file]);
  });

  it("normalizes unnamed files from clipboardData.files", () => {
    const file = new File(["x"], "", { type: "application/pdf" });
    const clipboard = createMockDataTransfer({ files: [file] });

    expect(getClipboardFiles(clipboard)[0].name).toMatch(
      /^pasted-image-\d+\.bin$/,
    );
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
      items: [
        {
          kind: "string",
          type: "text/plain",
          file: new File(["not a clipboard file"], "ignored.txt"),
        },
      ],
    });

    expect(getClipboardFiles(clipboard)).toEqual([]);
  });

  it("ignores file items that do not expose a File", () => {
    const clipboard = createMockDataTransfer({
      items: [{ kind: "file", type: "image/png", file: null }],
    });

    expect(getClipboardFiles(clipboard)).toEqual([]);
  });
});

describe("module-initialized clipboard contracts", () => {
  it("preserves file extensions, normalized names, and editable text", async () => {
    vi.resetModules();
    const fresh =
      await import("#/components/features/chat/utils/chat-input.utils");
    vi.spyOn(Date, "now").mockReturnValue(1234);

    for (const [type, extension] of [
      ["image/png", "png"],
      ["image/jpeg", "jpg"],
      ["image/jpg", "jpg"],
      ["image/gif", "gif"],
      ["image/webp", "webp"],
      ["image/bmp", "bmp"],
    ]) {
      expect(
        fresh.normalizePastedFile(new File(["x"], "", { type })).name,
      ).toBe(`pasted-image-1234.${extension}`);
    }

    const named = (name: string) =>
      fresh.isPastedClipboardImage(new File(["x"], name));
    expect(named("pasted-image-1710000000000.png")).toBe(true);
    expect(named("prefix-pasted-image-1710000000000.png")).toBe(false);
    expect(named("pasted-image-1710000000000.png.extra")).toBe(false);

    const editable = document.createElement("div");
    editable.innerText = "message";
    expect(fresh.getTextContent(editable)).toBe("message");
  });
});

describe("content-editable helpers", () => {
  it("detects empty elements across null, innerText, and textContent", () => {
    expect(isContentEmpty(null)).toBe(true);

    const element = document.createElement("div");
    element.innerText = "visible";
    element.textContent = "fallback";
    expect(isContentEmpty(element)).toBe(false);

    element.innerText = "";
    element.textContent = " fallback ";
    expect(isContentEmpty(element)).toBe(false);

    element.textContent = " \n ";
    expect(isContentEmpty(element)).toBe(true);

    element.innerText = "";
    element.textContent = "";
    expect(isContentEmpty(element)).toBe(true);
  });

  it("clears only empty editable content", () => {
    clearEmptyContent(null);
    const nonempty = document.createElement("div");
    nonempty.innerText = "keep";
    nonempty.innerHTML = "<b>keep</b>";
    clearEmptyContent(nonempty);
    expect(nonempty.innerHTML).toBe("<b>keep</b>");

    const empty = document.createElement("div");
    empty.innerText = " ";
    empty.innerHTML = "<br>";
    clearEmptyContent(empty);
    expect(empty.innerHTML).toBe("");
    expect(empty.textContent).toBe("");
  });

  it("gets and clears editable text and file inputs safely", () => {
    expect(getTextContent(null)).toBe("");
    const editable = document.createElement("div");
    editable.innerText = "message";
    expect(getTextContent(editable)).toBe("message");
    editable.textContent = "message";
    clearTextContent(editable);
    expect(editable.textContent).toBe("");
    clearTextContent(null);

    const input = document.createElement("input");
    input.value = "C:\\fakepath\\file.txt";
    clearFileInput(input);
    expect(input.value).toBe("");
    clearFileInput(null);
  });

  it("scrolls only when the cursor falls below the visible input", () => {
    const element = document.createElement("div");
    Object.defineProperties(element, {
      scrollHeight: { configurable: true, value: 200 },
      clientHeight: { configurable: true, value: 80 },
    });
    vi.spyOn(element, "getBoundingClientRect").mockReturnValue({
      bottom: 100,
    } as DOMRect);
    const range = {
      getBoundingClientRect: vi.fn(() => ({ bottom: 120 }) as DOMRect),
    } as unknown as Range;
    vi.spyOn(window, "getSelection").mockReturnValue({
      rangeCount: 1,
      getRangeAt: () => range,
    } as unknown as Selection);

    ensureCursorVisible(element);
    expect(element.scrollTop).toBe(120);

    range.getBoundingClientRect = vi.fn(() => ({ bottom: 90 }) as DOMRect);
    element.scrollTop = 0;
    ensureCursorVisible(element);
    expect(element.scrollTop).toBe(0);

    range.getBoundingClientRect = vi.fn(() => ({ bottom: 100 }) as DOMRect);
    ensureCursorVisible(element);
    expect(element.scrollTop).toBe(0);
  });

  it("returns safely when cursor geometry is unavailable", () => {
    const element = document.createElement("div");
    const selectionSpy = vi.spyOn(window, "getSelection");
    selectionSpy.mockReturnValue(null);
    ensureCursorVisible(element);

    selectionSpy.mockReturnValue({ rangeCount: 0 } as Selection);
    ensureCursorVisible(element);

    selectionSpy.mockReturnValue({
      rangeCount: 1,
      getRangeAt: () => ({}) as Range,
    } as unknown as Selection);
    ensureCursorVisible(element);

    selectionSpy.mockReturnValue({
      rangeCount: 1,
      getRangeAt: () =>
        ({ getBoundingClientRect: () => ({ bottom: 1 }) }) as Range,
    } as unknown as Selection);
    expect(() => ensureCursorVisible(null)).not.toThrow();
    Object.defineProperty(element, "getBoundingClientRect", {
      configurable: true,
      value: undefined,
    });
    ensureCursorVisible(element);
  });

  it("focuses content-editable text at the end", () => {
    focusContentEditableAtEnd(null);

    const element = document.createElement("div");
    element.textContent = "hello";
    document.body.appendChild(element);
    const focus = vi.spyOn(element, "focus");
    const selection = window.getSelection()!;
    focusContentEditableAtEnd(element);
    expect(focus).toHaveBeenCalledOnce();
    expect(selection.rangeCount).toBe(1);
    const caret = selection.getRangeAt(0);
    expect(caret.collapsed).toBe(true);
    expect(caret.startOffset).toBe(element.childNodes.length);

    vi.spyOn(window, "getSelection").mockReturnValue(null);
    focusContentEditableAtEnd(element);
    expect(focus).toHaveBeenCalledTimes(2);
  });
});
