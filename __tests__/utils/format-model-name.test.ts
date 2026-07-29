import { describe, expect, it } from "vitest";
import { formatNativeModelName } from "#/utils/format-model-name";

describe("formatNativeModelName", () => {
  it("strips the provider route prefix", () => {
    expect(formatNativeModelName("anthropic/claude-sonnet-4-5-20250929")).toBe(
      "claude-sonnet-4-5-20250929",
    );
    expect(formatNativeModelName("openai/gpt-4o")).toBe("gpt-4o");
  });

  it("strips nested routing prefixes to the last segment", () => {
    expect(formatNativeModelName("litellm_proxy/openai/gpt-4o")).toBe("gpt-4o");
  });

  it("returns the original string when there is no prefix", () => {
    expect(formatNativeModelName("gpt-4o")).toBe("gpt-4o");
  });

  it("falls back to the original string instead of returning empty (trailing slash)", () => {
    expect(formatNativeModelName("openai/")).toBe("openai/");
  });

  it("returns null for empty / nullish input", () => {
    expect(formatNativeModelName("")).toBeNull();
    expect(formatNativeModelName(null)).toBeNull();
    expect(formatNativeModelName(undefined)).toBeNull();
  });
});
