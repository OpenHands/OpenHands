import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { processFiles, processImages } from "#/utils/file-processing";

const reads = vi.hoisted(() => ({
  arrayBuffers: [] as string[],
  dataUrls: [] as string[],
}));

class ControlledFileReader {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  private finish(file: File) {
    queueMicrotask(() => {
      if (file.name.startsWith("error")) {
        this.onerror?.();
      } else if (file.name.startsWith("abort")) {
        this.onabort?.();
      } else {
        this.onload?.();
      }
    });
  }

  readAsArrayBuffer(file: File) {
    reads.arrayBuffers.push(file.name);
    this.finish(file);
  }

  readAsDataURL(file: File) {
    reads.dataUrls.push(file.name);
    this.finish(file);
  }
}

function file(name: string, type = "text/plain") {
  return new File(["content"], name, { type });
}

describe("file processing", () => {
  beforeEach(() => {
    reads.arrayBuffers = [];
    reads.dataUrls = [];
    vi.stubGlobal("FileReader", ControlledFileReader);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns successful files in input order while retaining read failures", async () => {
    const first = file("first.txt");
    const failed = file("error-report.txt");
    const last = file("last.txt");

    const result = await processFiles([first, failed, last]);

    expect(reads.arrayBuffers).toEqual([
      "first.txt",
      "error-report.txt",
      "last.txt",
    ]);
    expect(result.successful).toEqual([first, last]);
    expect(result.failed).toEqual([
      {
        file: failed,
        error: new Error("Failed to read file: error-report.txt"),
      },
    ]);
  });

  it("reports aborted regular-file reads", async () => {
    const aborted = file("abort-upload.txt");

    await expect(processFiles([aborted])).resolves.toEqual({
      successful: [],
      failed: [
        {
          file: aborted,
          error: new Error("File reading was aborted: abort-upload.txt"),
        },
      ],
    });
  });

  it("reads images as data URLs and isolates errors and aborts", async () => {
    const successful = file("photo.png", "image/png");
    const failed = file("error-photo.png", "image/png");
    const aborted = file("abort-photo.png", "image/png");

    const result = await processImages([successful, failed, aborted]);

    expect(reads.dataUrls).toEqual([
      "photo.png",
      "error-photo.png",
      "abort-photo.png",
    ]);
    expect(result.successful).toEqual([successful]);
    expect(result.failed).toEqual([
      {
        file: failed,
        error: new Error("Failed to read image: error-photo.png"),
      },
      {
        file: aborted,
        error: new Error("Image reading was aborted: abort-photo.png"),
      },
    ]);
  });

  it("handles empty file and image batches without creating readers", async () => {
    await expect(processFiles([])).resolves.toEqual({
      successful: [],
      failed: [],
    });
    await expect(processImages([])).resolves.toEqual({
      successful: [],
      failed: [],
    });
    expect(reads.arrayBuffers).toEqual([]);
    expect(reads.dataUrls).toEqual([]);
  });
});
