import { describe, it, expect } from "vitest";
import { formatLlmModel } from "#/utils/format-llm-model";

describe("formatLlmModel", () => {
  describe("provider prefix stripping", () => {
    it("strips a single provider prefix", () => {
      expect(formatLlmModel("openai/gpt-4o")).toBe("GPT-4o");
    });

    it("strips chained proxy prefixes (last segment wins)", () => {
      expect(
        formatLlmModel("litellm_proxy/anthropic/claude-3-5-sonnet-20241022"),
      ).toBe("Claude 3.5 Sonnet");
    });

    it("passes through unprefixed bare model names", () => {
      expect(formatLlmModel("my-finetune")).toBe("my-finetune");
    });
  });

  describe("date suffix stripping", () => {
    it("strips -YYYYMMDD suffix", () => {
      expect(formatLlmModel("anthropic/claude-sonnet-4-5-20250929")).toBe(
        "Claude Sonnet 4.5",
      );
    });

    it("strips -YYYY-MM-DD suffix", () => {
      expect(formatLlmModel("openai/gpt-4o-2024-08-06")).toBe("GPT-4o");
    });
  });

  describe("Claude family (4.x naming)", () => {
    it.each([
      ["anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5"],
      ["anthropic/claude-sonnet-4", "Claude Sonnet 4"],
      ["anthropic/claude-opus-4-7", "Claude Opus 4.7"],
      ["anthropic/claude-haiku-4-5", "Claude Haiku 4.5"],
    ])("formats %s", (raw, expected) => {
      expect(formatLlmModel(raw)).toBe(expected);
    });
  });

  describe("Claude family (3.x naming)", () => {
    it.each([
      ["anthropic/claude-3-5-sonnet", "Claude 3.5 Sonnet"],
      ["anthropic/claude-3-opus", "Claude 3 Opus"],
      ["anthropic/claude-3-haiku", "Claude 3 Haiku"],
    ])("formats %s", (raw, expected) => {
      expect(formatLlmModel(raw)).toBe(expected);
    });
  });

  describe("GPT family", () => {
    it.each([
      ["openai/gpt-4o", "GPT-4o"],
      ["openai/gpt-4o-mini", "GPT-4o mini"],
      ["openai/gpt-4.1", "GPT-4.1"],
      ["openai/gpt-5", "GPT-5"],
      ["openai/gpt-5-pro", "GPT-5 pro"],
    ])("formats %s", (raw, expected) => {
      expect(formatLlmModel(raw)).toBe(expected);
    });
  });

  describe("o-series", () => {
    it.each([
      ["openhands/o3", "o3"],
      ["openai/o3-mini", "o3-mini"],
      ["openai/o4-mini", "o4-mini"],
    ])("formats %s", (raw, expected) => {
      expect(formatLlmModel(raw)).toBe(expected);
    });
  });

  describe("Gemini family", () => {
    it.each([
      ["gemini/gemini-2.5-pro", "Gemini 2.5 Pro"],
      ["gemini/gemini-2.0-flash", "Gemini 2.0 Flash"],
      ["gemini/gemini-2.0-flash-001", "Gemini 2.0 Flash 001"],
    ])("formats %s", (raw, expected) => {
      expect(formatLlmModel(raw)).toBe(expected);
    });
  });

  describe("empty / fallback", () => {
    it("returns empty input unchanged", () => {
      expect(formatLlmModel("")).toBe("");
    });

    it("returns unknown models with prefix stripped", () => {
      expect(formatLlmModel("custom-provider/some-weird-model")).toBe(
        "some-weird-model",
      );
    });
  });
});
