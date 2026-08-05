import { describe, expect, it } from "vitest";
import { compareSemanticVersions } from "./version";

describe("compareSemanticVersions", () => {
  it("compares major, minor, and patch segments", () => {
    expect(compareSemanticVersions("1.2.1", "1.2.2")).toBe(-1);
    expect(compareSemanticVersions("1.3.0", "1.2.9")).toBe(1);
    expect(compareSemanticVersions("2.0.0", "1.99.99")).toBe(1);
  });

  it("treats omitted segments as zero", () => {
    expect(compareSemanticVersions("1.2", "1.2.0")).toBe(0);
    expect(compareSemanticVersions("1", "1.0.1")).toBe(-1);
  });

  it("accepts tags and build metadata from package feeds", () => {
    expect(compareSemanticVersions("v1.2.0", "1.2.0+build.1")).toBe(0);
    expect(compareSemanticVersions("1.2.0-beta.1", "1.2.1")).toBe(-1);
  });
});
