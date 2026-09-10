import { describe, expect, it } from "vitest";
import {
  ACP_PROVIDERS as CLIENT_ACP_PROVIDERS,
  getAcpProvider as getClientAcpProvider,
} from "@openhands/typescript-client";
import {
  ACP_MANAGED_SENTINEL,
  SURFACED_ACP_PROVIDERS,
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

describe("SURFACED_ACP_PROVIDERS and ACP provider surfacing", () => {
  it("surfaces exactly claude-code, codex, gemini-cli", () => {
    expect([...SURFACED_ACP_PROVIDERS]).toEqual([
      "claude-code",
      "codex",
      "gemini-cli",
    ]);
  });

  it("asserts that every surfaced provider exists in the pinned client registry", () => {
    for (const key of SURFACED_ACP_PROVIDERS) {
      const clientInfo = getClientAcpProvider(key);
      expect(clientInfo).toBeDefined();
      expect(clientInfo).not.toBeNull();
    }
  });

  it("breaks loudly if a surfaced provider is renamed or removed upstream", () => {
    const mockRegistryMissingClaude = Object.fromEntries(
      Object.entries(CLIENT_ACP_PROVIDERS).filter(
        ([key]) => key !== "claude-code",
      ),
    );
    expect(() => {
      for (const key of SURFACED_ACP_PROVIDERS) {
        if (!(key in mockRegistryMissingClaude)) {
          throw new Error(
            `Surfaced provider "${key}" not found in upstream registry`,
          );
        }
      }
    }).toThrow(
      'Surfaced provider "claude-code" not found in upstream registry',
    );
  });

  it("requires no change and passes unchanged when upstream adds new harnesses", () => {
    // Simulating SDK 1.45+ registering new harnesses (e.g. kimi-code, pi, opencode)
    const expandedRegistry = {
      ...CLIENT_ACP_PROVIDERS,
      "kimi-code": { key: "kimi-code", display_name: "Kimi" },
      pi: { key: "pi", display_name: "Pi" },
      opencode: { key: "opencode", display_name: "OpenCode" },
    };
    const allSurfacedPresent = SURFACED_ACP_PROVIDERS.every(
      (key) => key in expandedRegistry,
    );
    expect(allSurfacedPresent).toBe(true);
    // The surfaced list remains unpolluted
    expect(SURFACED_ACP_PROVIDERS).toHaveLength(3);
  });
});

describe("getAcpProviderSecrets", () => {
  it("produces credential fields for surfaced providers", () => {
    for (const key of SURFACED_ACP_PROVIDERS) {
      const secrets = getAcpProviderSecrets(key);
      expect(secrets.length).toBeGreaterThan(0);
    }
  });

  it("returns [] for any provider Canvas does not surface", () => {
    // Derives unsurfaced providers from the client registry (runs against any client version)
    const clientKeys = Object.keys(CLIENT_ACP_PROVIDERS);
    const unsurfacedKeys = clientKeys.filter(
      (key) => !(SURFACED_ACP_PROVIDERS as readonly string[]).includes(key),
    );
    for (const key of unsurfacedKeys) {
      expect(getAcpProviderSecrets(key)).toEqual([]);
    }

    // Explicit checks for known upstream harnesses not surfaced in Canvas
    expect(getAcpProviderSecrets("kimi-code")).toEqual([]);
    expect(getAcpProviderSecrets("pi")).toEqual([]);
    expect(getAcpProviderSecrets("opencode")).toEqual([]);
    expect(getAcpProviderSecrets("custom")).toEqual([]);
    expect(getAcpProviderSecrets("unknown-provider")).toEqual([]);
    expect(getAcpProviderSecrets(null)).toEqual([]);
    expect(getAcpProviderSecrets(undefined)).toEqual([]);
  });
});
