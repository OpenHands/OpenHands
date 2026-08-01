import { describe, expect, it } from "vitest";
import {
  satisfiesCanvasVersion,
  stripSecretValues,
  // eslint-disable-next-line import/no-relative-packages
} from "../scripts/skin-service.mjs";

describe("satisfiesCanvasVersion", () => {
  it("accepts any version when the range is empty or *", () => {
    expect(satisfiesCanvasVersion("1.8.0", "")).toBe(true);
    expect(satisfiesCanvasVersion("1.8.0", undefined)).toBe(true);
    expect(satisfiesCanvasVersion("1.8.0", "*")).toBe(true);
  });

  it("enforces >= lower bounds", () => {
    expect(satisfiesCanvasVersion("1.8.0", ">=1.7.0")).toBe(true);
    expect(satisfiesCanvasVersion("1.6.9", ">=1.7.0")).toBe(false);
  });

  it("enforces combined ranges", () => {
    expect(satisfiesCanvasVersion("1.8.0", ">=1.7.0 <2.0.0")).toBe(true);
    expect(satisfiesCanvasVersion("2.0.0", ">=1.7.0 <2.0.0")).toBe(false);
  });

  it("never blocks dev builds", () => {
    expect(satisfiesCanvasVersion("dev", ">=1.7.0")).toBe(true);
    expect(satisfiesCanvasVersion("unknown", ">=99.0.0")).toBe(true);
  });

  it("accepts v-prefixed versions", () => {
    expect(satisfiesCanvasVersion("v1.8.0", ">=1.7.0")).toBe(true);
  });
});

describe("stripSecretValues", () => {
  it("removes secret-ish string values but keeps names/structure", () => {
    const input = {
      name: "slack",
      api_key: "sk-live-abc123",
      token: "xoxb-secret",
      url: "https://example.com",
      nested: { password: "hunter2", model: "gpt-5" },
    };
    expect(stripSecretValues(input)).toEqual({
      name: "slack",
      url: "https://example.com",
      nested: { model: "gpt-5" },
    });
  });

  it("removes masked values returned by the agent server", () => {
    expect(stripSecretValues({ value: "**********", kept: "yes" })).toEqual({
      kept: "yes",
    });
  });

  it("handles arrays and scalars", () => {
    expect(stripSecretValues([{ token: "x", a: 1 }, "plain"])).toEqual([
      { a: 1 },
      "plain",
    ]);
    expect(stripSecretValues("s")).toBe("s");
    expect(stripSecretValues(null)).toBe(null);
  });
});
