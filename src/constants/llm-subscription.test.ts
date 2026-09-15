import { describe, expect, it } from "vitest";
import { matchSubscriptionModel } from "./llm-subscription";

describe("matchSubscriptionModel", () => {
  const serverModels = ["gpt-5.2-codex", "gpt-5.6-terra"];

  it("returns the id on an exact match", () => {
    expect(matchSubscriptionModel("gpt-5.6-terra", serverModels)).toBe(
      "gpt-5.6-terra",
    );
  });

  it("resolves an openai/-prefixed id to the canonical server id", () => {
    expect(matchSubscriptionModel("openai/gpt-5.6-terra", serverModels)).toBe(
      "gpt-5.6-terra",
    );
  });

  it("prefers the exact match when both forms are listed", () => {
    expect(
      matchSubscriptionModel("openai/gpt-5.6-terra", [
        "openai/gpt-5.6-terra",
        "gpt-5.6-terra",
      ]),
    ).toBe("openai/gpt-5.6-terra");
  });

  it("returns null when the model is not offered in any form", () => {
    expect(matchSubscriptionModel("openai/gpt-9", serverModels)).toBeNull();
    expect(matchSubscriptionModel("gpt-9", serverModels)).toBeNull();
  });

  it("does not strip a non-openai provider prefix", () => {
    expect(
      matchSubscriptionModel("anthropic/gpt-5.6-terra", serverModels),
    ).toBeNull();
  });

  it("strips the prefix case-sensitively and only once", () => {
    expect(
      matchSubscriptionModel("OpenAI/gpt-5.6-terra", serverModels),
    ).toBeNull();
    expect(
      matchSubscriptionModel("openai/openai/gpt-5.6-terra", serverModels),
    ).toBeNull();
  });

  it("returns null for non-string models or a missing list", () => {
    expect(matchSubscriptionModel(null, serverModels)).toBeNull();
    expect(matchSubscriptionModel(undefined, serverModels)).toBeNull();
    expect(matchSubscriptionModel("gpt-5.6-terra", undefined)).toBeNull();
  });
});
