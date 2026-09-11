import { describe, expect, it } from "vitest";
import { readProfileMcpRefs } from "#/constants/profile-scope";

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
