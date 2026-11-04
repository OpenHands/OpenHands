import { describe, expect, it } from "vitest";
import { sessionGeneration } from "#/utils/session-generation";

describe("sessionGeneration", () => {
  it("is stable for the same key and differs across keys", async () => {
    const first = await sessionGeneration("key-1");
    expect(first).toBe(await sessionGeneration("key-1"));
    expect(first).not.toBe(await sessionGeneration("key-2"));
  });

  it("maps missing credentials to null", async () => {
    expect(await sessionGeneration(null)).toBeNull();
    expect(await sessionGeneration(undefined)).toBeNull();
    expect(await sessionGeneration("")).toBeNull();
  });

  it("is a keyed digest, not a hash of the credential", async () => {
    const key = "super-secret-session-key";
    const generation = await sessionGeneration(key);
    expect(generation).toMatch(/^[0-9a-f]{32}$/);
    expect(generation).not.toContain(key);
  });

  it("does not reveal a guessable key to an offline attack", async () => {
    // A known-answer test: the tag must not equal a plain SHA-256 of the key,
    // which is what an unkeyed fingerprint would expose.
    const key = "password123";
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(key),
    );
    const unkeyed = Array.from(new Uint8Array(digest, 0, 16), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");

    expect(await sessionGeneration(key)).not.toBe(unkeyed);
  });
});
