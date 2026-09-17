import { describe, expect, it } from "vitest";
import { getAcpProvider as getClientAcpProvider } from "@openhands/typescript-client";
import {
  ACP_CUSTOM_PRESET_KEY,
  ACP_PROVIDERS,
  ACP_VERTEX_SAFE_MODEL,
  buildAcpAgentSettingsDiff,
  getAcpCredentialConflicts,
  getAcpPreferredDefaultModel,
  getAcpProvider,
  getAcpProviderDisplayName,
  getAcpProviderSecrets,
} from "#/constants/acp-providers";

describe("getAcpProviderDisplayName", () => {
  it("resolves the three built-in registry keys to their human names", () => {
    expect(getAcpProviderDisplayName("claude-code")).toBe("Claude Code");
    expect(getAcpProviderDisplayName("codex")).toBe("Codex");
    expect(getAcpProviderDisplayName("gemini-cli")).toBe("Gemini CLI");
  });

  it("returns null for the Custom-command preset so callers can fall back to the generic 'ACP' label", () => {
    // The custom preset has no canonical brand name — the registry
    // resolver intentionally returns null so the conversation card renders
    // ``CONVERSATION$ACP_AGENT_GENERIC`` ("ACP") instead.
    expect(getAcpProviderDisplayName("custom")).toBeNull();
  });

  it("returns null for unknown / forward-compatible keys", () => {
    // A future ACP server Canvas's registry doesn't know about yet
    // shouldn't crash or render a random fragment of the key — fall back
    // to the generic chip.
    expect(getAcpProviderDisplayName("future-acp-server")).toBeNull();
  });

  it("returns null for empty / null / undefined input", () => {
    expect(getAcpProviderDisplayName(null)).toBeNull();
    expect(getAcpProviderDisplayName(undefined)).toBeNull();
    expect(getAcpProviderDisplayName("")).toBeNull();
  });
});

describe("ACP provider registry", () => {
  it("sources display_name / default_command / models from the SDK, not a local mirror", () => {
    // Core invariant of agent-canvas#678: the registry data fields must come
    // straight from @openhands/typescript-client's getAcpProvider(), so the
    // Python SDK stays the single source of truth. Only the UI-only overlay
    // (icon + description_key) is layered on locally.
    for (const provider of ACP_PROVIDERS) {
      const sdk = getClientAcpProvider(provider.key);
      expect(sdk, provider.key).not.toBeNull();
      expect(provider.display_name).toBe(sdk!.display_name);
      expect(provider.default_command).toEqual([...sdk!.default_command]);
      expect(provider.available_models).toEqual(
        sdk!.available_models.map((m) => ({ id: m.id, label: m.label })),
      );
      expect(provider.default_model).toBe(sdk!.default_model ?? undefined);
      // UI-only overlay stays local.
      expect(provider.icon).toBeTruthy();
      expect(provider.description_key).toBeTruthy();
    }
  });

  it("keeps every built-in default model in the UX suggestions", () => {
    for (const provider of ACP_PROVIDERS) {
      expect(provider.default_model, provider.key).toBeTruthy();
      expect(provider.available_models, provider.key).toBeTruthy();
      expect(
        provider.available_models?.some(
          (model) => model.id === provider.default_model,
        ),
        provider.key,
      ).toBe(true);
    }
  });

  it("does not suggest generic default model placeholders", () => {
    // Model lists are SDK-owned (see ACP_PROVIDERS) — Canvas no longer hand-keeps
    // them. The claude-code registry intentionally offers an id ``default``
    // labeled "Default (recommended)", a legitimate, well-labeled choice. Guard
    // against genuinely empty ids and bare placeholder labels, not the qualified
    // "Default (recommended)" entry.
    for (const provider of ACP_PROVIDERS) {
      for (const model of provider.available_models ?? []) {
        expect(model.id.trim(), provider.key).toBeTruthy();
        expect(
          model.label.trim().toLowerCase(),
          `${provider.key}:${model.id}`,
        ).not.toBe("default");
      }
    }
  });

  it("seeds built-in ACP diffs with the provider's preferred default model", () => {
    // Preferred default = registry default everywhere except Gemini, where
    // the Vertex-safe override applies (see getAcpPreferredDefaultModel) —
    // EVERY default-model surface must agree on this, including this diff
    // builder's fallback.
    for (const provider of ACP_PROVIDERS) {
      expect(buildAcpAgentSettingsDiff(provider.key)).toMatchObject({
        agent_kind: "acp",
        acp_server: provider.key,
        acp_model: getAcpPreferredDefaultModel(provider.key),
      });
    }
    expect(buildAcpAgentSettingsDiff("gemini-cli")).toMatchObject({
      acp_model: ACP_VERTEX_SAFE_MODEL,
    });
  });

  it("keeps custom ACP diffs model-optional", () => {
    expect(buildAcpAgentSettingsDiff(ACP_CUSTOM_PRESET_KEY)).toMatchObject({
      agent_kind: "acp",
      acp_server: ACP_CUSTOM_PRESET_KEY,
      acp_model: null,
    });
  });
});

describe("getAcpProviderSecrets — containerized credentials", () => {
  // These are the credentials a fresh container (no host login) needs, sourced
  // from the validated container contract (agent-canvas#1013/#1014) — if a
  // refactor drops one, ACP auth in a container silently breaks, so assert
  // each provider's exact field set.
  it("collects the subscription cred, api key, then base URL — in that order — for Codex", () => {
    const names = getAcpProviderSecrets("codex").map((f) => f.name);
    expect(names).toEqual([
      "CODEX_AUTH_JSON",
      "OPENAI_API_KEY",
      "OPENAI_BASE_URL",
    ]);
  });

  it("collects the OAuth token + api key for Claude Code", () => {
    const names = getAcpProviderSecrets("claude-code").map((f) => f.name);
    expect(names).toEqual([
      "CLAUDE_CODE_OAUTH_TOKEN",
      "ANTHROPIC_API_KEY",
      "ANTHROPIC_BASE_URL",
    ]);
  });

  it("collects the Vertex SA JSON + project/location/flag for Gemini CLI", () => {
    const names = getAcpProviderSecrets("gemini-cli").map((f) => f.name);
    expect(names).toEqual([
      "GOOGLE_APPLICATION_CREDENTIALS_JSON",
      "GOOGLE_CLOUD_PROJECT",
      "GOOGLE_CLOUD_LOCATION",
      "GOOGLE_GENAI_USE_VERTEXAI",
      "GEMINI_API_KEY",
      "GEMINI_BASE_URL",
    ]);
  });

  it("renders file-content blobs as multiline secret fields", () => {
    // ``multiline`` also drives the orphaned-credential warning on backends
    // that can't materialise file secrets (cloud, agent-canvas#1016).
    const codexBlob = getAcpProviderSecrets("codex").find(
      (f) => f.name === "CODEX_AUTH_JSON",
    );
    expect(codexBlob).toMatchObject({ multiline: true, secret: true });

    const geminiBlob = getAcpProviderSecrets("gemini-cli").find(
      (f) => f.name === "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    );
    expect(geminiBlob).toMatchObject({ multiline: true, secret: true });
  });

  it("never marks the base URL as a credential (not secret, not multiline)", () => {
    // ``secret`` is what a required credential step counts as an actual
    // credential — a base URL alone can't authenticate, and ANTHROPIC_BASE_URL
    // alongside a Claude OAuth token actively breaks bearer auth.
    for (const key of ["codex", "claude-code", "gemini-cli"]) {
      const baseUrl = getAcpProviderSecrets(key).find((f) =>
        f.name.endsWith("_BASE_URL"),
      );
      expect(baseUrl?.secret, key).toBeFalsy();
      expect(baseUrl?.multiline, key).toBeFalsy();
    }
  });

  it("returns [] for OpenHands / custom / unknown / empty", () => {
    expect(getAcpProviderSecrets("openhands")).toEqual([]);
    expect(getAcpProviderSecrets(ACP_CUSTOM_PRESET_KEY)).toEqual([]);
    expect(getAcpProviderSecrets("future-acp-server")).toEqual([]);
    expect(getAcpProviderSecrets(null)).toEqual([]);
  });
});

describe("getAcpPreferredDefaultModel", () => {
  it("overrides Gemini with the Vertex-safe model rather than the registry default", () => {
    // gemini-cli's own default 404s on many Vertex projects; canvas preselects
    // a broadly-available model instead.
    expect(getAcpPreferredDefaultModel("gemini-cli")).toBe(
      ACP_VERTEX_SAFE_MODEL,
    );
    expect(getAcpPreferredDefaultModel("gemini-cli")).not.toBe(
      getAcpProvider("gemini-cli")?.default_model,
    );
  });

  it("pins a NON-flash Gemini model", () => {
    // gemini-cli 0.45.x re-resolves any *-flash id at generation time to its
    // current default flash (software-agent-sdk#3532), so a flash pin is not
    // honored — only a non-flash id (e.g. gemini-2.5-pro) sticks.
    expect(ACP_VERTEX_SAFE_MODEL).not.toMatch(/flash/);
  });

  it("keeps the registry default for the other providers", () => {
    expect(getAcpPreferredDefaultModel("codex")).toBe(
      getAcpProvider("codex")?.default_model,
    );
    expect(getAcpPreferredDefaultModel("claude-code")).toBe(
      getAcpProvider("claude-code")?.default_model,
    );
  });

  it("returns null for OpenHands / custom / unknown", () => {
    expect(getAcpPreferredDefaultModel("openhands")).toBeNull();
    expect(getAcpPreferredDefaultModel(ACP_CUSTOM_PRESET_KEY)).toBeNull();
    expect(getAcpPreferredDefaultModel("future-acp-server")).toBeNull();
  });
});

describe("getAcpCredentialConflicts", () => {
  const has =
    (...names: string[]) =>
    (name: string) =>
      names.includes(name);

  it("flags the Claude OAuth token + base URL pair when both are set", () => {
    expect(
      getAcpCredentialConflicts(
        "claude-code",
        has("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"),
      ),
    ).toEqual([["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"]]);
  });

  it("flags the Claude OAuth token + API key pair when both are set", () => {
    // The SDK strips ANTHROPIC_API_KEY when the OAuth token is active
    // (software-agent-sdk#3588), so the key would be silently ignored.
    expect(
      getAcpCredentialConflicts(
        "claude-code",
        has("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
      ),
    ).toEqual([["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]]);
  });

  it("flags both pairs when the token, API key, and base URL are all set", () => {
    expect(
      getAcpCredentialConflicts(
        "claude-code",
        has(
          "CLAUDE_CODE_OAUTH_TOKEN",
          "ANTHROPIC_API_KEY",
          "ANTHROPIC_BASE_URL",
        ),
      ),
    ).toEqual([
      ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
      ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"],
    ]);
  });

  it("stays quiet when only one side is set", () => {
    expect(
      getAcpCredentialConflicts("claude-code", has("CLAUDE_CODE_OAUTH_TOKEN")),
    ).toEqual([]);
    expect(
      getAcpCredentialConflicts("claude-code", has("ANTHROPIC_BASE_URL")),
    ).toEqual([]);
  });

  it("has no conflicts for other providers / null", () => {
    expect(
      getAcpCredentialConflicts(
        "codex",
        has("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"),
      ),
    ).toEqual([]);
    expect(getAcpCredentialConflicts(null, () => true)).toEqual([]);
  });
});
