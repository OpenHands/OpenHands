import { describe, expect, it } from "vitest";
import {
  FREE_OPENHANDS_MODELS,
  formatModelNameForDisplay,
  formatNativeModelName,
  formatProviderModelNameForDisplay,
  isFreeOpenHandsModel,
} from "#/utils/format-model-name";

describe("formatNativeModelName", () => {
  it("strips the provider route prefix", () => {
    expect(formatNativeModelName("anthropic/claude-sonnet-4-5-20250929")).toBe(
      "claude-sonnet-4-5-20250929",
    );
    expect(formatNativeModelName("openai/gpt-4o")).toBe("gpt-4o");
  });

  it("labels only configured OpenHands free-model routes as free", () => {
    expect(Object.keys(FREE_OPENHANDS_MODELS)).toEqual([
      "openhands/glm-5.2",
      "openhands/deepseek-v4-flash",
      "openhands/minimax-m2.7",
    ]);

    for (const [model, label] of Object.entries(FREE_OPENHANDS_MODELS)) {
      expect(formatModelNameForDisplay(model)).toBe(label);
      expect(
        formatProviderModelNameForDisplay("openhands", model.slice(10)),
      ).toBe(label);
      expect(isFreeOpenHandsModel(model)).toBe(true);
    }

    expect(formatModelNameForDisplay("openai/glm-5.2")).toBe("openai/glm-5.2");
    expect(formatProviderModelNameForDisplay("openai", "glm-5.2")).toBe(
      "glm-5.2",
    );
    expect(isFreeOpenHandsModel("openai/glm-5.2")).toBe(false);
  });

  it("keeps free OpenHands labels on native conversation chips", () => {
    expect(formatNativeModelName("openhands/glm-5.2")).toBe(
      FREE_OPENHANDS_MODELS["openhands/glm-5.2"],
    );
    expect(formatNativeModelName("openhands/deepseek-v4-flash")).toBe(
      FREE_OPENHANDS_MODELS["openhands/deepseek-v4-flash"],
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
