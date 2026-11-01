import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  acpModelScope,
  readRememberedAcpModels,
  rememberAcpModels,
  subscribeRememberedAcpModels,
} from "#/utils/remembered-acp-models";

const MODELS = [{ id: "sonnet", label: "Sonnet" }];
const ALL = acpModelScope(null);

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("remembered ACP models", () => {
  it("notifies subscribers when a list changes, not when it is rewritten unchanged", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeRememberedAcpModels(listener);

    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);
    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    rememberAcpModels("cloud-1", "claude-code", ALL, [
      { id: "haiku", label: "Haiku" },
    ]);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("keeps each backend's list per agent", () => {
    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);

    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
    expect(readRememberedAcpModels("cloud-2", "claude-code", ALL)).toEqual([]);
    expect(readRememberedAcpModels("cloud-1", "codex", ALL)).toEqual([]);
  });

  it("keeps a list per secret scope", () => {
    const narrow = acpModelScope(["OPENAI_API_KEY"]);
    const haiku = [{ id: "haiku", label: "Haiku" }];
    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);
    rememberAcpModels("cloud-1", "claude-code", narrow, haiku);

    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
    expect(readRememberedAcpModels("cloud-1", "claude-code", narrow)).toEqual(
      haiku,
    );
    expect(
      readRememberedAcpModels("cloud-1", "claude-code", acpModelScope([])),
    ).toEqual([]);
  });

  it("names a scope by its secrets, whatever their order", () => {
    expect(acpModelScope(["B", "A", "B"])).toBe(acpModelScope(["A", "B"]));
    expect(acpModelScope([])).not.toBe(acpModelScope(null));
  });

  it("offers every scope's models when no scope is given", () => {
    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);
    rememberAcpModels("cloud-1", "claude-code", acpModelScope([]), [
      { id: "sonnet", label: "Sonnet (again)" },
      { id: "haiku", label: "Haiku" },
    ]);

    expect(readRememberedAcpModels("cloud-1", "claude-code")).toEqual([
      ...MODELS,
      { id: "haiku", label: "Haiku" },
    ]);
  });

  it("does not replace a list with an empty one", () => {
    rememberAcpModels("cloud-1", "claude-code", ALL, MODELS);
    rememberAcpModels("cloud-1", "claude-code", ALL, []);

    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
  });

  it("drops entries that are not models", () => {
    window.localStorage.setItem(
      "openhands-acp-models:cloud-1:claude-code",
      JSON.stringify({ "*": [...MODELS, { id: "" }, "haiku", null] }),
    );

    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual(
      MODELS,
    );
  });

  it("returns nothing when storage cannot be read", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(readRememberedAcpModels("cloud-1", "claude-code", ALL)).toEqual([]);
  });
});
