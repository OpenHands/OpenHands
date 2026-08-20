import { describe, expect, it } from "vitest";
import {
  PROFILE_NAME_PATTERN,
  deriveProfileNameFromModel,
} from "#/utils/derive-profile-name";

describe("PROFILE_NAME_PATTERN", () => {
  it("matches valid profile names starting with alphanumeric", () => {
    expect(PROFILE_NAME_PATTERN.test("a")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("Z")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("0")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("9")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("gpt-4")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("claude_3.5")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("my-profile-name")).toBe(true);
    expect(PROFILE_NAME_PATTERN.test("Profile_With.Mixed-Chars")).toBe(true);
  });

  it("rejects empty strings", () => {
    expect(PROFILE_NAME_PATTERN.test("")).toBe(false);
  });

  it("rejects names starting with non-alphanumeric characters", () => {
    expect(PROFILE_NAME_PATTERN.test(".hidden")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("_underscore")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("-dash")).toBe(false);
  });

  it("rejects names with invalid characters", () => {
    expect(PROFILE_NAME_PATTERN.test("name with spaces")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("name/slash")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("name\\backslash")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("name:colon")).toBe(false);
    expect(PROFILE_NAME_PATTERN.test("name@at")).toBe(false);
  });

  it("rejects names longer than 64 characters", () => {
    const exactlyValid = "a".repeat(64);
    const tooLong = "a".repeat(65);
    expect(PROFILE_NAME_PATTERN.test(exactlyValid)).toBe(true);
    expect(PROFILE_NAME_PATTERN.test(tooLong)).toBe(false);
  });
});

describe("deriveProfileNameFromModel", () => {
  it("extracts model name after provider prefix", () => {
    expect(deriveProfileNameFromModel("openai/gpt-4")).toBe("gpt-4");
    expect(deriveProfileNameFromModel("anthropic/claude-3")).toBe("claude-3");
    expect(deriveProfileNameFromModel("google/gemini-pro")).toBe("gemini-pro");
  });

  it("handles nested provider paths", () => {
    expect(deriveProfileNameFromModel("together/meta-llama/Llama-3")).toBe(
      "Llama-3",
    );
    expect(deriveProfileNameFromModel("azure/openai/gpt-4-turbo")).toBe(
      "gpt-4-turbo",
    );
  });

  it("returns model string as-is if no slash present", () => {
    expect(deriveProfileNameFromModel("gpt-4")).toBe("gpt-4");
    expect(deriveProfileNameFromModel("claude")).toBe("claude");
  });

  it("sanitizes invalid characters to dashes", () => {
    expect(deriveProfileNameFromModel("provider/model with spaces")).toBe(
      "model-with-spaces",
    );
    expect(deriveProfileNameFromModel("provider/model:variant")).toBe(
      "model-variant",
    );
  });

  it("collapses consecutive dashes", () => {
    expect(deriveProfileNameFromModel("provider/model--name")).toBe(
      "model-name",
    );
    expect(deriveProfileNameFromModel("provider/model@@@name")).toBe(
      "model-name",
    );
  });

  it("removes leading and trailing dashes", () => {
    expect(deriveProfileNameFromModel("provider/-model-")).toBe("model");
    expect(deriveProfileNameFromModel("provider/--name--")).toBe("name");
  });

  it("prepends 'profile-' when result starts with non-alphanumeric", () => {
    expect(deriveProfileNameFromModel("provider/.hidden")).toBe(
      "profile-.hidden",
    );
    expect(deriveProfileNameFromModel("provider/_underscore")).toBe(
      "profile-_underscore",
    );
  });

  it("truncates result to 64 characters", () => {
    const longModelName = "a".repeat(100);
    const result = deriveProfileNameFromModel(`provider/${longModelName}`);
    expect(result.length).toBeLessThanOrEqual(64);
    expect(result).toBe("a".repeat(64));
  });

  it("removes trailing dashes after truncation", () => {
    // Create a model name that when truncated at 64 chars ends with a dash
    // 63 'a's + dash at position 64 + more chars that get truncated
    const model = "a".repeat(63) + "-" + "b".repeat(10);
    const result = deriveProfileNameFromModel(`provider/${model}`);
    expect(result.length).toBe(63); // Trailing dash removed
    expect(result.endsWith("-")).toBe(false);
    expect(result).toBe("a".repeat(63));
  });

  it("returns 'default-profile' for empty input", () => {
    expect(deriveProfileNameFromModel("")).toBe("default-profile");
  });

  it("returns 'default-profile' when model name sanitizes to empty string", () => {
    // Model names that only contain invalid characters
    expect(deriveProfileNameFromModel("provider/@@@")).toBe("default-profile");
    expect(deriveProfileNameFromModel("provider/---")).toBe("default-profile");
    expect(deriveProfileNameFromModel("provider/!!!")).toBe("default-profile");
    expect(deriveProfileNameFromModel("provider/   ")).toBe("default-profile");
  });

  it("handles trailing slash by falling back to full input", () => {
    // "provider/" splits to ["provider", ""], falls back to "provider/"
    // which sanitizes "/" to "-" then trims to "provider"
    expect(deriveProfileNameFromModel("provider/")).toBe("provider");
    // "/" splits to ["", ""], falls back to "/" which sanitizes to empty,
    // then returns "default-profile"
    expect(deriveProfileNameFromModel("/")).toBe("default-profile");
  });
});
