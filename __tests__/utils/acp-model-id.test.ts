import { describe, expect, it } from "vitest";
import {
  composeAcpModelId,
  getAcpEffortLevels,
  parseAcpModelId,
} from "#/utils/acp-model-id";

describe("parseAcpModelId", () => {
  it("splits a claude-code id on a recognized effort suffix", () => {
    expect(parseAcpModelId("sonnet/high", "claude-code")).toEqual({
      base: "sonnet",
      effort: "high",
    });
  });

  it("recognizes every claude-code effort level", () => {
    for (const effort of ["low", "medium", "high", "xhigh", "max"]) {
      expect(parseAcpModelId(`sonnet/${effort}`, "claude-code")).toEqual({
        base: "sonnet",
        effort,
      });
    }
  });

  it("recognizes every codex effort level", () => {
    for (const effort of ["low", "medium", "high", "xhigh"]) {
      expect(parseAcpModelId(`gpt-5.1-codex/${effort}`, "codex")).toEqual({
        base: "gpt-5.1-codex",
        effort,
      });
    }
  });

  it("does not split codex on 'max' (not a codex effort level)", () => {
    // "max" is valid for claude-code but not codex — the check must be
    // per-server, not a global effort-level set.
    expect(parseAcpModelId("gpt-5.1-codex/max", "codex")).toEqual({
      base: "gpt-5.1-codex/max",
      effort: null,
    });
  });

  it("does not split on an unrecognized suffix", () => {
    expect(parseAcpModelId("sonnet/turbo", "claude-code")).toEqual({
      base: "sonnet/turbo",
      effort: null,
    });
  });

  it("never splits gemini-cli ids", () => {
    expect(parseAcpModelId("gemini-2.5-pro/high", "gemini-cli")).toEqual({
      base: "gemini-2.5-pro/high",
      effort: null,
    });
  });

  it("never splits the custom preset's ids", () => {
    expect(parseAcpModelId("my-model/high", "custom")).toEqual({
      base: "my-model/high",
      effort: null,
    });
  });

  it("never splits for an unknown/unregistered server", () => {
    expect(parseAcpModelId("sonnet/high", "some-future-server")).toEqual({
      base: "sonnet/high",
      effort: null,
    });
  });

  it("never splits when no server is known (null/undefined)", () => {
    expect(parseAcpModelId("sonnet/high", null)).toEqual({
      base: "sonnet/high",
      effort: null,
    });
    expect(parseAcpModelId("sonnet/high", undefined)).toEqual({
      base: "sonnet/high",
      effort: null,
    });
  });

  it("leaves an id with no slash untouched", () => {
    expect(parseAcpModelId("sonnet", "claude-code")).toEqual({
      base: "sonnet",
      effort: null,
    });
  });

  it("splits only on the LAST slash", () => {
    expect(parseAcpModelId("vertex/sonnet-4-5/high", "claude-code")).toEqual({
      base: "vertex/sonnet-4-5",
      effort: "high",
    });
  });

  it("does not split when the base would be empty (id is just '/<effort>')", () => {
    expect(parseAcpModelId("/high", "claude-code")).toEqual({
      base: "/high",
      effort: null,
    });
  });

  it("leaves an empty id untouched", () => {
    expect(parseAcpModelId("", "claude-code")).toEqual({
      base: "",
      effort: null,
    });
  });
});

describe("composeAcpModelId", () => {
  it("composes a base + recognized effort into the '<base>/<effort>' id", () => {
    expect(composeAcpModelId("sonnet", "high", "claude-code")).toBe(
      "sonnet/high",
    );
    expect(composeAcpModelId("gpt-5.1-codex", "medium", "codex")).toBe(
      "gpt-5.1-codex/medium",
    );
  });

  it("composes every recognized level for claude-code and codex", () => {
    for (const effort of ["low", "medium", "high", "xhigh", "max"]) {
      expect(composeAcpModelId("sonnet", effort, "claude-code")).toBe(
        `sonnet/${effort}`,
      );
    }
    for (const effort of ["low", "medium", "high", "xhigh"]) {
      expect(composeAcpModelId("gpt-5.1-codex", effort, "codex")).toBe(
        `gpt-5.1-codex/${effort}`,
      );
    }
  });

  it("returns the bare base for null/undefined/empty effort", () => {
    expect(composeAcpModelId("sonnet", null, "claude-code")).toBe("sonnet");
    expect(composeAcpModelId("sonnet", undefined, "claude-code")).toBe(
      "sonnet",
    );
    expect(composeAcpModelId("sonnet", "", "claude-code")).toBe("sonnet");
  });

  it('returns the bare base for the "default" sentinel', () => {
    expect(composeAcpModelId("sonnet", "default", "claude-code")).toBe(
      "sonnet",
    );
    expect(composeAcpModelId("gpt-5.5", "default", "codex")).toBe("gpt-5.5");
  });

  it("returns the bare base when the level isn't valid for that server", () => {
    // "max" is a claude-code level, not a codex one.
    expect(composeAcpModelId("gpt-5.1-codex", "max", "codex")).toBe(
      "gpt-5.1-codex",
    );
    // Not a recognized level at all.
    expect(composeAcpModelId("sonnet", "turbo", "claude-code")).toBe("sonnet");
  });

  it("never composes for gemini-cli, the custom preset, or an unknown/no server", () => {
    expect(composeAcpModelId("gemini-2.5-pro", "high", "gemini-cli")).toBe(
      "gemini-2.5-pro",
    );
    expect(composeAcpModelId("my-model", "high", "custom")).toBe("my-model");
    expect(composeAcpModelId("sonnet", "high", "some-future-server")).toBe(
      "sonnet",
    );
    expect(composeAcpModelId("sonnet", "high", null)).toBe("sonnet");
    expect(composeAcpModelId("sonnet", "high", undefined)).toBe("sonnet");
  });

  it("round-trips through parseAcpModelId for every recognized level", () => {
    for (const [server, levels] of [
      ["claude-code", ["low", "medium", "high", "xhigh", "max"]],
      ["codex", ["low", "medium", "high", "xhigh"]],
    ] as const) {
      for (const effort of levels) {
        const composed = composeAcpModelId("sonnet", effort, server);
        expect(parseAcpModelId(composed, server)).toEqual({
          base: "sonnet",
          effort,
        });
      }
    }
  });

  it('round-trips "default"/null back to a bare base parseAcpModelId never splits', () => {
    expect(
      parseAcpModelId(
        composeAcpModelId("sonnet", "default", "claude-code"),
        "claude-code",
      ),
    ).toEqual({ base: "sonnet", effort: null });
    expect(
      parseAcpModelId(
        composeAcpModelId("sonnet", null, "claude-code"),
        "claude-code",
      ),
    ).toEqual({ base: "sonnet", effort: null });
  });
});

describe("getAcpEffortLevels", () => {
  it("lists claude-code's UI-selectable levels with 'default' first", () => {
    expect(getAcpEffortLevels("claude-code")).toEqual([
      "default",
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]);
  });

  it("lists codex's UI-selectable levels with 'default' first (no 'max')", () => {
    expect(getAcpEffortLevels("codex")).toEqual([
      "default",
      "low",
      "medium",
      "high",
      "xhigh",
    ]);
  });

  it("returns null for gemini-cli, the custom preset, and an unknown/no server", () => {
    expect(getAcpEffortLevels("gemini-cli")).toBeNull();
    expect(getAcpEffortLevels("custom")).toBeNull();
    expect(getAcpEffortLevels("some-future-server")).toBeNull();
    expect(getAcpEffortLevels(null)).toBeNull();
    expect(getAcpEffortLevels(undefined)).toBeNull();
  });
});
