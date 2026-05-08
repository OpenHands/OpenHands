import { describe, it, expect } from "vitest";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";

describe("deriveProfileNameFromModel", () => {
  it("replaces the provider slash with an underscore", () => {
    expect(deriveProfileNameFromModel("openai/gpt-4o")).toBe("openai_gpt-4o");
    expect(deriveProfileNameFromModel("anthropic/claude-3-5-sonnet")).toBe(
      "anthropic_claude-3-5-sonnet",
    );
  });

  it("collapses disallowed characters into single underscores", () => {
    // Colons, spaces, plus signs, etc. are not in [A-Za-z0-9._-].
    expect(deriveProfileNameFromModel("openai/gpt-4o:custom")).toBe(
      "openai_gpt-4o_custom",
    );
    expect(deriveProfileNameFromModel("a b c")).toBe("a_b_c");
    expect(deriveProfileNameFromModel("a+++b")).toBe("a_b");
  });

  it("preserves dots, dashes, and underscores as-is", () => {
    expect(deriveProfileNameFromModel("foo.bar-baz_qux")).toBe(
      "foo.bar-baz_qux",
    );
  });

  it("trims leading/trailing separator characters", () => {
    expect(deriveProfileNameFromModel("  ___openai/gpt-4o___  ")).toBe(
      "openai_gpt-4o",
    );
  });

  it("returns null for empty or unusable input", () => {
    expect(deriveProfileNameFromModel("")).toBeNull();
    expect(deriveProfileNameFromModel("   ")).toBeNull();
    expect(deriveProfileNameFromModel("///")).toBeNull();
    expect(deriveProfileNameFromModel("!!!")).toBeNull();
  });

  it("truncates at 64 characters to respect the backend cap", () => {
    const long = "a".repeat(200);
    const result = deriveProfileNameFromModel(long);
    expect(result).not.toBeNull();
    expect(result!.length).toBe(64);
  });

  it("passes through an already-valid name unchanged", () => {
    expect(deriveProfileNameFromModel("my-profile")).toBe("my-profile");
    expect(deriveProfileNameFromModel("foo_bar.baz-1")).toBe("foo_bar.baz-1");
  });

  it("leaves names exactly 64 characters long untouched", () => {
    const name = "a".repeat(64);
    expect(deriveProfileNameFromModel(name)).toBe(name);
  });

  it("truncates names of 65+ characters down to 64", () => {
    const over = "a".repeat(65);
    const result = deriveProfileNameFromModel(over);
    expect(result).not.toBeNull();
    expect(result!.length).toBe(64);
  });

  it("produces names that satisfy the backend regex", () => {
    const pattern = /^[A-Za-z0-9._-]{1,64}$/;
    for (const model of [
      "openai/gpt-4o",
      "anthropic/claude-3-5-sonnet",
      "openhands/claude-sonnet-4-20250514",
      "azure/standard/1024-x-1024/dall-e-2",
      "openai/gpt-4o:custom",
    ]) {
      const name = deriveProfileNameFromModel(model);
      expect(name).not.toBeNull();
      expect(pattern.test(name!)).toBe(true);
    }
  });
});
