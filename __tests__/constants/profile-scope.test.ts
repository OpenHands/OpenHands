import { describe, expect, it } from "vitest";
import {
  readProfileMcpRefs,
  sameScopeSelection,
} from "#/constants/profile-scope";

describe("readProfileMcpRefs", () => {
  it("reads an absent field as the server default", () => {
    expect(readProfileMcpRefs(undefined)).toEqual({
      mode: "standard",
      selected: [],
    });
    expect(readProfileMcpRefs(null)).toEqual({
      mode: "standard",
      selected: [],
    });
  });

  it("reads an empty array as an explicit no-servers scope", () => {
    expect(readProfileMcpRefs([])).toEqual({ mode: "custom", selected: [] });
  });

  it("reads a list as the selection", () => {
    expect(readProfileMcpRefs(["github", "postgres"])).toEqual({
      mode: "custom",
      selected: ["github", "postgres"],
    });
  });

  it("drops non-string entries rather than failing the editor", () => {
    expect(readProfileMcpRefs(["github", 7, null])).toEqual({
      mode: "custom",
      selected: ["github"],
    });
  });
});

describe("sameScopeSelection", () => {
  it("ignores order, which the resolver does not use", () => {
    expect(
      sameScopeSelection(["github", "postgres"], ["postgres", "github"]),
    ).toBe(true);
  });

  it("ignores a repeated name, which the resolver collapses", () => {
    expect(sameScopeSelection(["github"], ["github", "github"])).toBe(true);
  });

  it("still sees an added or removed server", () => {
    expect(sameScopeSelection(["github"], ["github", "postgres"])).toBe(false);
    expect(sameScopeSelection(["github"], ["postgres"])).toBe(false);
  });

  it("treats an empty selection as equal only to another empty one", () => {
    expect(sameScopeSelection([], [])).toBe(true);
    expect(sameScopeSelection([], ["github"])).toBe(false);
  });
});
