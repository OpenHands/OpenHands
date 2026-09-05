import { describe, expect, it } from "vitest";
import {
  buildRouterModel,
  collectRequiredRouterModelNames,
  parseModelTableNames,
} from "#/components/features/settings/meta-llm-profiles/router-profiles";

describe("parseModelTableNames", () => {
  it("extracts the leading token of each list row and ignores descriptions", () => {
    const table = [
      "- GPT-5.4: swe-bench: 75.60%/$0.63; gaia: 82.40%",
      "- MiniMax-M3 efficient fallback",
      "not a list line",
      "- claude-opus-4-8",
    ].join("\n");
    expect(parseModelTableNames(table)).toEqual([
      "GPT-5.4",
      "MiniMax-M3",
      "claude-opus-4-8",
    ]);
  });

  it("de-duplicates while preserving first-seen order", () => {
    expect(parseModelTableNames("- a x\n- b y\n- a z")).toEqual(["a", "b"]);
  });

  it("de-duplicates case-insensitively while preserving the first spelling", () => {
    expect(parseModelTableNames("- MiniMax-M3 x\n- minimax-m3 y")).toEqual([
      "MiniMax-M3",
    ]);
  });

  it("returns an empty list for empty or missing tables", () => {
    expect(parseModelTableNames("")).toEqual([]);
    expect(parseModelTableNames(null)).toEqual([]);
    expect(parseModelTableNames(undefined)).toEqual([]);
  });
});

describe("buildRouterModel", () => {
  it("lower-cases the provider and prefixes the model name", () => {
    expect(buildRouterModel("OpenHands", "GPT-5.4")).toBe("openhands/GPT-5.4");
    expect(buildRouterModel("anthropic", "claude-opus-4-8")).toBe(
      "anthropic/claude-opus-4-8",
    );
  });
});

describe("collectRequiredRouterModelNames", () => {
  it("combines table names with classifier and default, de-duplicated", () => {
    expect(
      collectRequiredRouterModelNames({
        classifier_model: "MiniMax-M3",
        default_model: "router-default",
        model_table: "- GPT-5.4 stats\n- MiniMax-M3 stats",
      }),
    ).toEqual(["GPT-5.4", "MiniMax-M3", "router-default"]);
  });

  it("de-duplicates classifier/default against table names case-insensitively", () => {
    expect(
      collectRequiredRouterModelNames({
        classifier_model: "minimax-m3",
        default_model: "ROUTER-DEFAULT",
        model_table: "- MiniMax-M3 stats\n- router-default stats",
      }),
    ).toEqual(["MiniMax-M3", "router-default"]);
  });

  it("ignores blank classifier/default and empty tables", () => {
    expect(
      collectRequiredRouterModelNames({
        classifier_model: "",
        default_model: null,
        model_table: null,
      }),
    ).toEqual([]);
  });
});
