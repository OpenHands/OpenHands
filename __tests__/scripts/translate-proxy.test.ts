// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  chunkTextForTranslation,
  parseTranslateRequestBody,
  resolveTranslateTarget,
} from "../../scripts/translate-proxy.mjs";

describe("resolveTranslateTarget", () => {
  it("maps Portuguese UI locales to pt-BR", () => {
    expect(resolveTranslateTarget("pt")).toBe("pt-BR");
    expect(resolveTranslateTarget("pt-BR")).toBe("pt-BR");
  });

  it("skips English and empty languages", () => {
    expect(resolveTranslateTarget("en")).toBeNull();
    expect(resolveTranslateTarget("en-US")).toBeNull();
    expect(resolveTranslateTarget("")).toBeNull();
  });
});

describe("chunkTextForTranslation", () => {
  it("keeps short text as a single chunk", () => {
    expect(chunkTextForTranslation("hello world", 20)).toEqual(["hello world"]);
  });

  it("splits long text on word boundaries", () => {
    const text = "alpha beta gamma delta epsilon";
    expect(chunkTextForTranslation(text, 14)).toEqual([
      "alpha beta",
      "gamma delta",
      "epsilon",
    ]);
  });
});

describe("parseTranslateRequestBody", () => {
  it("accepts a valid Portuguese batch", () => {
    expect(
      parseTranslateRequestBody({
        texts: ["Detected use of eval", "  ", "Insecure pattern"],
        target: "pt",
      }),
    ).toEqual({
      texts: ["Detected use of eval", "Insecure pattern"],
      source: "en",
      target: "pt-BR",
    });
  });

  it("rejects English targets and non-array texts", () => {
    expect(() =>
      parseTranslateRequestBody({ texts: ["x"], target: "en" }),
    ).toThrow(/english/i);
    expect(() =>
      parseTranslateRequestBody({ texts: "x", target: "pt" }),
    ).toThrow(/array/i);
  });
});
