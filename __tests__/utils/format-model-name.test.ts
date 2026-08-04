import { describe, expect, it } from "vitest";
import {
  formatModelNameForDisplay,
  formatNativeModelName,
  formatProviderModelNameForDisplay,
  OPENHANDS_FREE_GLM_MODEL_LABEL,
} from "#/utils/format-model-name";

describe("formatNativeModelName", () => {
  it("strips the provider route prefix", () => {
    expect(formatNativeModelName("anthropic/claude-sonnet-4-5-20250929")).toBe(
      "claude-sonnet-4-5-20250929",
    );
    expect(formatNativeModelName("openai/gpt-4o")).toBe("gpt-4o");
  });

  it("labels only the OpenHands GLM-5.2 route as free", () => {
    expect(formatModelNameForDisplay("openhands/glm-5.2")).toBe(
      OPENHANDS_FREE_GLM_MODEL_LABEL,
    );
    expect(formatProviderModelNameForDisplay("openhands", "glm-5.2")).toBe(
      OPENHANDS_FREE_GLM_MODEL_LABEL,
    );
    expect(formatModelNameForDisplay("openai/glm-5.2")).toBe("openai/glm-5.2");
    expect(formatProviderModelNameForDisplay("openai", "glm-5.2")).toBe(
      "glm-5.2",
    );
  });

  it("keeps the free OpenHands GLM-5.2 label on native conversation chips", () => {
    expect(formatNativeModelName("openhands/glm-5.2")).toBe(
      OPENHANDS_FREE_GLM_MODEL_LABEL,
    );
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
