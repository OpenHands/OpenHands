import { describe, expect, it } from "vitest";
import {
  buildRouterModel,
  collectRequiredRouterModelNames,
  parseModelTableNames,
} from "./router-profiles";

const MODEL_TABLE = `- gpt-5.4: stats
- minimax-m3: stats`;

describe("router-profiles", () => {
  describe("collectRequiredRouterModelNames", () => {
    it("includes the classifier model alongside the routable table models", () => {
      expect(
        collectRequiredRouterModelNames({
          classifier_model: "minimax-m3",
          model_table: MODEL_TABLE,
        }),
      ).toEqual(["gpt-5.4", "minimax-m3"]);
    });

    it("includes a classifier model that is not in the table (creation path)", () => {
      expect(
        collectRequiredRouterModelNames({
          classifier_model: "classifier-x",
          model_table: MODEL_TABLE,
        }),
      ).toEqual(["gpt-5.4", "minimax-m3", "classifier-x"]);
    });

    it("ignores a blank classifier", () => {
      expect(
        collectRequiredRouterModelNames({
          classifier_model: "",
          model_table: MODEL_TABLE,
        }),
      ).toEqual(["gpt-5.4", "minimax-m3"]);
    });
  });

  describe("buildRouterModel", () => {
    it("derives the router endpoint from the provider connection", () => {
      expect(buildRouterModel("openai", "minimax-m3")).toBe(
        "openai/minimax-m3",
      );
    });
  });

  describe("parseModelTableNames", () => {
    it("parses bullet-prefixed model names and strips trailing colons", () => {
      expect(parseModelTableNames(MODEL_TABLE)).toEqual([
        "gpt-5.4",
        "minimax-m3",
      ]);
    });
  });
});
