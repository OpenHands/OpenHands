import { afterEach, describe, expect, it, vi } from "vitest";
import { processFiles, processImages } from "#/utils/file-processing";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("attachment processing", () => {
  it("does not read selected blobs into browser memory", async () => {
    const fileReader = vi.fn(() => {
      throw new Error("attachments must not be read before upload");
    });
    vi.stubGlobal("FileReader", fileReader);
    const files = [new File(["content"], "design.pdf")];
    const images = [
      new File(["content"], "diagram.png", { type: "image/png" }),
    ];

    await expect(processFiles(files)).resolves.toEqual({
      successful: files,
      failed: [],
    });
    await expect(processImages(images)).resolves.toEqual({
      successful: images,
      failed: [],
    });
    expect(fileReader).not.toHaveBeenCalled();
  });
});
