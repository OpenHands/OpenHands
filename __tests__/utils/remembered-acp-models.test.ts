import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  readRememberedAcpModels,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

const MODELS = [{ id: "sonnet", label: "Sonnet" }];

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("remembered ACP models", () => {
  it("keeps each backend's list per agent", () => {
    rememberAcpModels("cloud-1", "claude-code", MODELS);

    expect(readRememberedAcpModels("cloud-1", "claude-code")).toEqual(MODELS);
    expect(readRememberedAcpModels("cloud-2", "claude-code")).toEqual([]);
    expect(readRememberedAcpModels("cloud-1", "codex")).toEqual([]);
  });

  it("does not replace a list with an empty one", () => {
    rememberAcpModels("cloud-1", "claude-code", MODELS);
    rememberAcpModels("cloud-1", "claude-code", []);

    expect(readRememberedAcpModels("cloud-1", "claude-code")).toEqual(MODELS);
  });

  it("drops entries that are not models", () => {
    window.localStorage.setItem(
      "openhands-acp-models:cloud-1:claude-code",
      JSON.stringify([...MODELS, { id: "" }, "haiku", null]),
    );

    expect(readRememberedAcpModels("cloud-1", "claude-code")).toEqual(MODELS);
  });

  it("returns nothing when storage cannot be read", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(readRememberedAcpModels("cloud-1", "claude-code")).toEqual([]);
  });
});
