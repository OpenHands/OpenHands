import { describe, expect, it } from "vitest";
import { sessionGeneration } from "#/utils/session-generation";

describe("sessionGeneration", () => {
  it("is stable for the same key and differs across keys", () => {
    expect(sessionGeneration("key-1")).toBe(sessionGeneration("key-1"));
    expect(sessionGeneration("key-1")).not.toBe(sessionGeneration("key-2"));
  });

  it("maps missing credentials to null", () => {
    expect(sessionGeneration(null)).toBeNull();
    expect(sessionGeneration(undefined)).toBeNull();
    expect(sessionGeneration("")).toBeNull();
  });

  it("does not contain the credential", () => {
    const key = "super-secret-session-key";
    const generation = sessionGeneration(key);
    expect(generation).not.toBeNull();
    expect(generation).not.toContain(key);
    expect(key).not.toContain(generation as string);
  });
});
