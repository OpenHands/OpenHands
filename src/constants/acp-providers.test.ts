import { describe, expect, it } from "vitest";
import {
  ACP_MANAGED_SENTINEL,
  ACP_PROVIDERS,
  resolveCodexDefaultCommand,
  resolveEffectiveAcpModel,
} from "./acp-providers";

describe("ACP_PROVIDERS", () => {
  it("offers GPT-6 Astra first for Codex ACP", () => {
    const codex = ACP_PROVIDERS.find(({ key }) => key === "codex");
    expect(codex?.available_models?.[0]).toEqual({
      id: "gpt-6-astra",
      label: "GPT-6 Astra",
    });
    expect(codex?.default_command).toEqual([
      "npx",
      "-y",
      "@agentclientprotocol/codex-acp@1.10.0",
    ]);
  });
});

describe("resolveCodexDefaultCommand", () => {
  it("overrides only the exact stale codex pin from the pinned client", () => {
    expect(
      resolveCodexDefaultCommand([
        "npx",
        "-y",
        "@agentclientprotocol/codex-acp@1.1.7",
      ]),
    ).toEqual(["npx", "-y", "@agentclientprotocol/codex-acp@1.10.0"]);
    expect(
      resolveCodexDefaultCommand([
        "npx",
        "-y",
        "@agentclientprotocol/codex-acp@1.10.0",
      ]),
    ).toEqual(["npx", "-y", "@agentclientprotocol/codex-acp@1.10.0"]);
    expect(
      resolveCodexDefaultCommand([
        "npx",
        "-y",
        "@agentclientprotocol/codex-acp@2.0.0",
      ]),
    ).toEqual(["npx", "-y", "@agentclientprotocol/codex-acp@2.0.0"]);
  });
});

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
