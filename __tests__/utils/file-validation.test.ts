import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getAttachmentLimits,
  validateEmbeddedImageSizes,
  validateFiles,
} from "#/utils/file-validation";

const MB = 1024 * 1024;
const RUNTIME_LIMITS_KEY = "__AGENT_CANVAS_ATTACHMENT_LIMITS__";

function makeFile(name: string, size: number, type = "text/plain") {
  const file = new File([""], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function setRuntimeLimits(value: unknown) {
  (window as unknown as Record<string, unknown>)[RUNTIME_LIMITS_KEY] = value;
}

afterEach(() => {
  vi.unstubAllEnvs();
  delete (window as unknown as Record<string, unknown>)[RUNTIME_LIMITS_KEY];
});

describe("attachment file validation", () => {
  it("accepts regular files larger than the previous 3MB limit", () => {
    expect(validateFiles([makeFile("design.pdf", 10 * MB)])).toEqual({
      isValid: true,
    });
  });

  it("enforces the default per-file and aggregate limits", () => {
    expect(validateFiles([makeFile("archive.zip", 25 * MB)]).isValid).toBe(
      true,
    );

    const oversized = validateFiles([makeFile("archive.zip", 25 * MB + 1)]);
    expect(oversized).toMatchObject({
      isValid: false,
      oversizedFiles: ["archive.zip"],
    });
    expect(oversized.errorMessage).toContain("25MB");

    expect(
      validateFiles(
        [makeFile("new.zip", 20 * MB)],
        [makeFile("existing.zip", 30 * MB)],
      ).isValid,
    ).toBe(true);

    const overTotal = validateFiles(
      [makeFile("new.zip", 20 * MB + 1)],
      [makeFile("existing.zip", 30 * MB)],
    );
    expect(overTotal.isValid).toBe(false);
    expect(overTotal.errorMessage).toContain("50MB limit");
  });

  it("uses build-time overrides and keeps the total at least the per-file limit", () => {
    vi.stubEnv("VITE_MAX_ATTACHMENT_FILE_SIZE_MB", "12.5");
    vi.stubEnv("VITE_MAX_ATTACHMENT_TOTAL_SIZE_MB", "8");

    expect(getAttachmentLimits()).toEqual({
      maxFileSizeMb: 12.5,
      maxTotalSizeMb: 12.5,
    });
    expect(validateFiles([makeFile("data.csv", 12 * MB)]).isValid).toBe(true);
  });

  it("prefers valid runtime overrides and ignores malformed values", () => {
    vi.stubEnv("VITE_MAX_ATTACHMENT_FILE_SIZE_MB", "9");
    vi.stubEnv("VITE_MAX_ATTACHMENT_TOTAL_SIZE_MB", "15");
    setRuntimeLimits({ maxFileSizeMb: 20, maxTotalSizeMb: "40" });

    expect(getAttachmentLimits()).toEqual({
      maxFileSizeMb: 20,
      maxTotalSizeMb: 40,
    });

    setRuntimeLimits({ maxFileSizeMb: -1, maxTotalSizeMb: "invalid" });
    expect(getAttachmentLimits()).toEqual({
      maxFileSizeMb: 25,
      maxTotalSizeMb: 50,
    });
  });
});

describe("embedded image validation", () => {
  it("retains the 3MB inline-image cap while allowing upload as a file", () => {
    const image = makeFile("diagram.png", 4 * MB, "image/png");

    expect(validateFiles([image]).isValid).toBe(true);
    expect(validateEmbeddedImageSizes([image])).toMatchObject({
      isValid: false,
      oversizedFiles: ["diagram.png"],
    });
  });

  it("accepts an embedded image exactly at the limit", () => {
    expect(
      validateEmbeddedImageSizes([
        makeFile("screenshot.png", 3 * MB, "image/png"),
      ]),
    ).toEqual({ isValid: true });
  });
});
