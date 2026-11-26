import { describe, expect, it } from "vitest";
import {
  resolveUploadLimits,
  validateFiles,
  validateIndividualFileSizes,
  validateTotalFileSize,
} from "#/utils/file-validation";

const MEBIBYTE = 1024 * 1024;
const THREE_MEBIBYTES = 3 * MEBIBYTE;

function fileWithSize(
  name: string,
  size: number,
  type = "application/octet-stream",
) {
  const file = new File([], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function imageWithSize(name: string, size: number) {
  return fileWithSize(name, size, "image/png");
}

describe("attachment size validation", () => {
  it("accepts a file at the default limit and rejects one byte over it", () => {
    expect(
      validateIndividualFileSizes([
        fileWithSize("at-limit.zip", THREE_MEBIBYTES),
      ]),
    ).toEqual({ isValid: true });

    expect(
      validateIndividualFileSizes([
        fileWithSize("over-limit.zip", THREE_MEBIBYTES + 1),
      ]),
    ).toEqual({
      isValid: false,
      errorMessage: "Files exceeding 3MB are not allowed: over-limit.zip",
      oversizedFiles: ["over-limit.zip"],
    });
  });

  it("reports every oversized file by name in selection order", () => {
    const result = validateIndividualFileSizes([
      fileWithSize("first.bin", THREE_MEBIBYTES + 1),
      fileWithSize("allowed.bin", THREE_MEBIBYTES),
      fileWithSize("second.bin", 4 * MEBIBYTE),
    ]);

    expect(result).toEqual({
      isValid: false,
      errorMessage:
        "Files exceeding 3MB are not allowed: first.bin, second.bin",
      oversizedFiles: ["first.bin", "second.bin"],
    });
  });

  it("accepts new and existing files whose combined size is exactly the limit", () => {
    expect(
      validateTotalFileSize(
        [fileWithSize("new.bin", MEBIBYTE)],
        [fileWithSize("existing.bin", 2 * MEBIBYTE)],
      ),
    ).toEqual({ isValid: true });
  });

  it("rejects a new and existing aggregate one byte over the limit", () => {
    expect(
      validateTotalFileSize(
        [fileWithSize("new.bin", MEBIBYTE + 1)],
        [fileWithSize("existing.bin", 2 * MEBIBYTE)],
      ),
    ).toEqual({
      isValid: false,
      errorMessage:
        "Total file size would be 3.0MB, exceeding the 3MB limit. Please select fewer or smaller files.",
    });
  });

  it("formats an oversized aggregate to one decimal place", () => {
    expect(
      validateTotalFileSize(
        [fileWithSize("new.bin", 2 * MEBIBYTE)],
        [fileWithSize("existing.bin", 1.25 * MEBIBYTE)],
      ),
    ).toEqual({
      isValid: false,
      errorMessage:
        "Total file size would be 3.3MB, exceeding the 3MB limit. Please select fewer or smaller files.",
    });
  });

  it("uses an empty existing-file list when none is provided", () => {
    expect(
      validateTotalFileSize([
        fileWithSize("new.bin", THREE_MEBIBYTES + MEBIBYTE),
      ]),
    ).toEqual({
      isValid: false,
      errorMessage:
        "Total file size would be 4.0MB, exceeding the 3MB limit. Please select fewer or smaller files.",
    });
  });

  it("returns the individual-file error before the aggregate-size error", () => {
    expect(
      validateFiles(
        [fileWithSize("too-large.bin", THREE_MEBIBYTES + 1)],
        [fileWithSize("existing.bin", MEBIBYTE)],
      ),
    ).toEqual({
      isValid: false,
      errorMessage: "Files exceeding 3MB are not allowed: too-large.bin",
      oversizedFiles: ["too-large.bin"],
    });
  });

  it("checks the aggregate after every new file passes individually", () => {
    expect(
      validateFiles([
        fileWithSize("first.bin", 2 * MEBIBYTE),
        fileWithSize("second.bin", 2 * MEBIBYTE),
      ]),
    ).toEqual({
      isValid: false,
      errorMessage:
        "Total file size would be 4.0MB, exceeding the 3MB limit. Please select fewer or smaller files.",
    });
  });

  it("accepts an aggregate at the limit through the combined validator", () => {
    expect(
      validateFiles(
        [fileWithSize("new.bin", MEBIBYTE)],
        [fileWithSize("existing.bin", 2 * MEBIBYTE)],
      ),
    ).toEqual({ isValid: true });
  });

  it("holds an image to the provider's limit rather than the default one", () => {
    const limits = resolveUploadLimits("anthropic");

    // 4 MB is over the 3 MB default but within Anthropic's 5 MB.
    expect(
      validateIndividualFileSizes(
        [imageWithSize("shot.png", 4 * MEBIBYTE)],
        limits,
      ),
    ).toEqual({ isValid: true });

    expect(
      validateIndividualFileSizes(
        [imageWithSize("shot.png", 6 * MEBIBYTE)],
        limits,
      ),
    ).toEqual({
      isValid: false,
      errorMessage: "Files exceeding 5MB are not allowed: shot.png",
      oversizedFiles: ["shot.png"],
    });
  });

  it("holds a non-image attachment to the default limit even on a permissive provider", () => {
    const limits = resolveUploadLimits("openai");

    // The provider never receives the bytes of a PDF, so its 50 MB image
    // limit says nothing about this attachment.
    expect(
      validateIndividualFileSizes(
        [fileWithSize("report.pdf", 4 * MEBIBYTE, "application/pdf")],
        limits,
      ),
    ).toEqual({
      isValid: false,
      errorMessage: "Files exceeding 3MB are not allowed: report.pdf",
      oversizedFiles: ["report.pdf"],
    });
  });

  it("keeps an image and a file on different ceilings in one selection", () => {
    const limits = resolveUploadLimits("anthropic");

    const result = validateIndividualFileSizes(
      [
        imageWithSize("shot.png", 6 * MEBIBYTE),
        fileWithSize("report.pdf", 4 * MEBIBYTE, "application/pdf"),
      ],
      limits,
    );

    expect(result.isValid).toBe(false);
    expect(result.errorMessage).toContain(
      "Files exceeding 5MB are not allowed: shot.png",
    );
    expect(result.errorMessage).toContain(
      "Files exceeding 3MB are not allowed: report.pdf",
    );
    expect(result.oversizedFiles).toEqual(["shot.png", "report.pdf"]);
  });

  it("falls back to the default limit for a provider with no published limit", () => {
    const limits = resolveUploadLimits("some-unknown-provider");

    expect(
      validateIndividualFileSizes(
        [imageWithSize("shot.png", 4 * MEBIBYTE)],
        limits,
      ),
    ).toEqual({
      isValid: false,
      errorMessage: "Files exceeding 3MB are not allowed: shot.png",
      oversizedFiles: ["shot.png"],
    });
  });

  it("resolves the default limits when no provider is known", () => {
    expect(resolveUploadLimits(null)).toEqual({
      imageLimitBytes: 3 * MEBIBYTE,
      imageLimitMb: 3,
      attachmentLimitBytes: 3 * MEBIBYTE,
      attachmentLimitMb: 3,
      provider: null,
    });
  });
});
