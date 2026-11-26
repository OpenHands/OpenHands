import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_ATTACHMENT_LIMIT_MB,
  MAX_EMBEDDED_IMAGE_BYTES,
  getProviderImageLimitBytes,
  getProviderImageLimitMb,
  hasProviderImageLimit,
  isImageAttachment,
  listProviderImageLimits,
} from "#/utils/provider-upload-limits";

const MEBIBYTE = 1024 * 1024;

describe("provider upload limits", () => {
  it("reports each documented provider limit", () => {
    expect(getProviderImageLimitMb("anthropic")).toBe(5);
    expect(getProviderImageLimitMb("gemini")).toBe(20);
    expect(getProviderImageLimitMb("openai")).toBe(50);
    expect(getProviderImageLimitMb("azure")).toBe(50);
    expect(getProviderImageLimitMb("vertex_ai")).toBe(20);
  });

  it("falls back to the default limit for a provider with no published limit", () => {
    expect(getProviderImageLimitMb("some-unknown-provider")).toBe(
      DEFAULT_ATTACHMENT_LIMIT_MB,
    );
    expect(getProviderImageLimitMb(null)).toBe(DEFAULT_ATTACHMENT_LIMIT_MB);
    expect(getProviderImageLimitMb(undefined)).toBe(
      DEFAULT_ATTACHMENT_LIMIT_MB,
    );
  });

  it("converts the limit to bytes", () => {
    expect(getProviderImageLimitBytes("anthropic")).toBe(5 * MEBIBYTE);
    expect(getProviderImageLimitBytes("unknown")).toBe(
      DEFAULT_ATTACHMENT_LIMIT_MB * MEBIBYTE,
    );
  });

  it("distinguishes a documented limit from the fallback", () => {
    expect(hasProviderImageLimit("anthropic")).toBe(true);
    expect(hasProviderImageLimit("some-unknown-provider")).toBe(false);
    expect(hasProviderImageLimit(null)).toBe(false);
  });

  it("lists every provider with a documented limit", () => {
    const listed = listProviderImageLimits();
    expect(listed).toContainEqual({ provider: "anthropic", limitMb: 5 });
    expect(listed).toContainEqual({ provider: "gemini", limitMb: 20 });
    expect(listed).toContainEqual({ provider: "openai", limitMb: 50 });
    // Every listed provider resolves to its own listed limit.
    for (const { provider, limitMb } of listed) {
      expect(getProviderImageLimitMb(provider)).toBe(limitMb);
    }
  });

  it("keeps the embedded-image ceiling at the transport bound", () => {
    // 4 MiB WebSocket frame minus ~4/3 base64 inflation.
    expect(MAX_EMBEDDED_IMAGE_BYTES).toBe(3 * MEBIBYTE);
    expect(MAX_EMBEDDED_IMAGE_BYTES).toBeLessThan(
      getProviderImageLimitBytes("openai"),
    );
  });

  it("classifies an image by its mime type", () => {
    expect(
      isImageAttachment(new File([], "a.png", { type: "image/png" })),
    ).toBe(true);
    expect(
      isImageAttachment(new File([], "a.pdf", { type: "application/pdf" })),
    ).toBe(false);
    expect(
      isImageAttachment(new File([], "a.mp4", { type: "video/mp4" })),
    ).toBe(false);
    expect(isImageAttachment(new File([], "a.bin"))).toBe(false);
  });

  describe("deployment overrides", () => {
    const WINDOW_KEY = "__AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS__";

    afterEach(() => {
      delete (window as unknown as Record<string, unknown>)[WINDOW_KEY];
    });

    it("uses the injected override for a provider", () => {
      (window as unknown as Record<string, unknown>)[WINDOW_KEY] =
        '{"anthropic":10}';

      expect(getProviderImageLimitMb("anthropic")).toBe(10);
      expect(hasProviderImageLimit("anthropic")).toBe(true);
    });

    it("leaves other providers on their documented limit", () => {
      (window as unknown as Record<string, unknown>)[WINDOW_KEY] =
        '{"anthropic":10}';

      expect(getProviderImageLimitMb("gemini")).toBe(20);
    });

    it("ignores an unparsable or malformed override", () => {
      (window as unknown as Record<string, unknown>)[WINDOW_KEY] = "not json";
      expect(getProviderImageLimitMb("anthropic")).toBe(5);

      (window as unknown as Record<string, unknown>)[WINDOW_KEY] =
        '{"anthropic":"ten"}';
      expect(getProviderImageLimitMb("anthropic")).toBe(5);

      (window as unknown as Record<string, unknown>)[WINDOW_KEY] =
        '{"anthropic":-1}';
      expect(getProviderImageLimitMb("anthropic")).toBe(5);
    });
  });
});
