import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  acpLaunchKey,
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
  describe("acpLaunchKey", () => {
    it("preserves providerKey for built-in providers without command overrides", () => {
      expect(acpLaunchKey("claude-code")).toBe("claude-code");
      expect(acpLaunchKey("codex", null, null)).toBe("codex");
      expect(acpLaunchKey("pi", [], [])).toBe("pi");
    });

    it("returns null for custom profile with empty or missing command", () => {
      expect(acpLaunchKey("custom")).toBeNull();
      expect(acpLaunchKey("custom", [])).toBeNull();
      expect(acpLaunchKey("custom", null)).toBeNull();
      expect(acpLaunchKey(null)).toBeNull();
      expect(acpLaunchKey(undefined)).toBeNull();
    });

    it("generates distinct keys for distinct commands or args", () => {
      const keyA = acpLaunchKey("custom", ["my-agent-a"]);
      const keyB = acpLaunchKey("custom", ["my-agent-b"]);
      const keyC = acpLaunchKey("custom", ["my-agent-a"], ["--flag"]);

      expect(keyA).not.toBeNull();
      expect(keyB).not.toBeNull();
      expect(keyC).not.toBeNull();
      expect(keyA).not.toBe(keyB);
      expect(keyA).not.toBe(keyC);
    });
  });

  it("isolates remembered models across different launch configurations", () => {
    const keyA = acpLaunchKey("custom", ["agent-a"])!;
    const keyB = acpLaunchKey("custom", ["agent-b"])!;
    const modelsA = [{ id: "model-a", label: "Model A" }];
    const modelsB = [{ id: "model-b", label: "Model B" }];

    rememberAcpModels("local-1", keyA, ALL, modelsA);
    rememberAcpModels("local-1", keyB, ALL, modelsB);

    expect(readRememberedAcpModels("local-1", keyA, ALL)).toEqual(modelsA);
    expect(readRememberedAcpModels("local-1", keyB, ALL)).toEqual(modelsB);
  });

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
