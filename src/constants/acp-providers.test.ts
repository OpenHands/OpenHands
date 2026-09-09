import { ACP_PROVIDERS as CLIENT_ACP_PROVIDERS } from "@openhands/typescript-client";
import { describe, expect, it } from "vitest";
import {
  ACP_MANAGED_SENTINEL,
  ACP_PROVIDER_NOT_SURFACED,
  ACP_PROVIDERS,
  getAcpProviderSecrets,
  resolveEffectiveAcpModel,
} from "./acp-providers";

describe("resolveEffectiveAcpModel", () => {
  it("surfaces the real claude-agent-acp 0.44+ 'default' model", () => {
    // ``default`` ("Default (recommended)") is a real, selectable Claude model
    // in the configOptions select — the server reports it as the current model.
    // It must NOT be suppressed as a placeholder (regression: the chip would
    // otherwise show no model for a session genuinely running on 'default').
    expect(resolveEffectiveAcpModel({ runtimeId: "default" })).toBe("default");
    expect(
      resolveEffectiveAcpModel({ runtimeName: "Default (recommended)" }),
    ).toBe("Default (recommended)");
  });

  it("follows the runtime → configured → sdkLlm precedence", () => {
    expect(
      resolveEffectiveAcpModel({
        runtimeName: "Sonnet",
        runtimeId: "sonnet",
        configured: "haiku",
      }),
    ).toBe("Sonnet");
    expect(resolveEffectiveAcpModel({ configured: "haiku" })).toBe("haiku");
    expect(resolveEffectiveAcpModel({ sdkLlm: "gpt-5.5/medium" })).toBe(
      "gpt-5.5/medium",
    );
  });

  it("still suppresses the legacy acp-managed sentinel and blanks", () => {
    expect(
      resolveEffectiveAcpModel({ sdkLlm: ACP_MANAGED_SENTINEL }),
    ).toBeNull();
    expect(resolveEffectiveAcpModel({ runtimeId: "   " })).toBeNull();
    expect(resolveEffectiveAcpModel({})).toBeNull();
  });

  it("falls back to providerDefault only when no concrete model resolves", () => {
    expect(
      resolveEffectiveAcpModel({
        sdkLlm: ACP_MANAGED_SENTINEL,
        providerDefault: "opus[1m]",
      }),
    ).toBe("opus[1m]");
    // A real 'default' wins over providerDefault — it is a concrete model.
    expect(
      resolveEffectiveAcpModel({
        runtimeId: "default",
        providerDefault: "opus[1m]",
      }),
    ).toBe("default");
  });
});

describe("surfaced ACP providers", () => {
  const surfaced = ACP_PROVIDERS.map(({ key }) => key);
  const notSurfaced = Object.keys(ACP_PROVIDER_NOT_SURFACED);

  it("makes an explicit decision about every registry key", () => {
    // Fails when @openhands/typescript-client is bumped past a harness Canvas
    // has not triaged, instead of half-wiring it into the picker.
    const decided = new Set([...surfaced, ...notSurfaced]);
    const undecided = Object.keys(CLIENT_ACP_PROVIDERS).filter(
      (key) => !decided.has(key),
    );

    expect(undecided).toEqual([]);
  });

  it("surfaces nothing the pinned client registry has dropped", () => {
    // The other direction: a rename or removal upstream must break loudly
    // rather than leave a tile whose command and models resolve to nothing.
    expect(surfaced.filter((key) => !(key in CLIENT_ACP_PROVIDERS))).toEqual(
      [],
    );
  });

  it("surfaces only Claude Code, Codex and Gemini CLI", () => {
    expect(surfaced).toEqual(["claude-code", "codex", "gemini-cli"]);
  });

  it("offers no credential fields for a provider it does not surface", () => {
    notSurfaced.forEach((key) => {
      expect(getAcpProviderSecrets(key)).toEqual([]);
    });
  });
});
