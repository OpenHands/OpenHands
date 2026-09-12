import { beforeEach, describe, expect, it } from "vitest";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

const BASE_URL = "http://localhost:3000";

const jsonRequest = (body: unknown, method: "PATCH" | "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const getJson = async <T>(path: string, init?: RequestInit) => {
  const response = await fetch(`${BASE_URL}${path}`, init);
  return { response, body: (await response.json()) as T };
};

beforeEach(() => {
  resetTestHandlersMockSettings();
});

describe("mock settings schemas and state", () => {
  it("serves both supported versions of each settings schema", async () => {
    for (const path of [
      "/api/settings/agent-schema",
      "/api/v1/settings/agent-schema",
    ]) {
      const { response, body } = await getJson<{
        model_name: string;
        sections: { key: string }[];
      }>(path);
      expect(response.ok).toBe(true);
      expect(body.model_name).toBe("AgentSettings");
      expect(body.sections.map(({ key }) => key)).toEqual([
        "general",
        "llm",
        "verification",
        "condenser",
        "agent_context",
      ]);
    }

    for (const path of [
      "/api/settings/conversation-schema",
      "/api/v1/settings/conversation-schema",
    ]) {
      const { response, body } = await getJson<{
        model_name: string;
        sections: { key: string }[];
      }>(path);
      expect(response.ok).toBe(true);
      expect(body).toMatchObject({
        model_name: "ConversationSettings",
        sections: [{ key: "general" }, { key: "verification" }],
      });
    }
  });

  it("returns deterministic defaults from both settings APIs", async () => {
    const modern = await getJson<{
      agent_settings: { llm: { model: string } };
      conversation_settings: { confirmation_mode: boolean };
      llm_api_key_is_set: boolean;
      misc_settings: { app_preferences: Record<string, unknown> };
    }>("/api/settings");
    expect(modern.response.ok).toBe(true);
    expect(modern.body).toMatchObject({
      agent_settings: { llm: { model: "openai/gpt-5.6-sol" } },
      conversation_settings: { confirmation_mode: false },
      llm_api_key_is_set: false,
      misc_settings: {
        app_preferences: {
          language: null,
          user_consents_to_analytics: null,
          enable_sound_notifications: null,
          git_user_name: null,
          git_user_email: null,
          disabled_skills: [],
        },
      },
    });

    const legacy = await getJson<{ agent_settings: unknown }>(
      "/api/v1/settings",
    );
    expect(legacy.response.ok).toBe(true);
    expect(legacy.body.agent_settings).toEqual(
      expect.objectContaining({
        llm: expect.objectContaining({ model: "openai/gpt-5.6-sol" }),
      }),
    );
  });

  it("rejects empty and no-op incremental updates", async () => {
    const nullBody = await getJson<{ error: string }>(
      "/api/settings",
      jsonRequest(null, "PATCH"),
    );
    expect(nullBody.response.status).toBe(400);
    expect(nullBody.body).toEqual({ error: "Empty body" });

    for (const body of [
      {},
      {
        agent_settings_diff: null,
        conversation_settings_diff: null,
        misc_settings_diff: null,
      },
    ]) {
      const result = await getJson<{ error: string }>(
        "/api/settings",
        jsonRequest(body, "PATCH"),
      );
      expect(result.response.status).toBe(400);
      expect(result.body.error).toContain("At least one of");
    }
  });

  it("deep-merges agent settings and persists all supported diff groups", async () => {
    const update = await getJson<{
      agent_settings: Record<string, unknown>;
      conversation_settings: Record<string, unknown>;
      llm_api_key_is_set: boolean;
      misc_settings: { app_preferences: Record<string, unknown> };
    }>(
      "/api/settings",
      jsonRequest(
        {
          agent_settings_diff: {
            llm: {
              api_key: "sk-secret",
              base_url: "https://api.example.test/v1",
              stop: ["DONE"],
              parameters: { reasoning: "high" },
            },
            verification: { critic_enabled: true },
            mcp_config: null,
          },
          conversation_settings_diff: {
            max_iterations: 12,
            confirmation_mode: true,
          },
          misc_settings_diff: {
            app_preferences: {
              language: "fr",
              disabled_skills: ["legacy-skill"],
            },
          },
        },
        "PATCH",
      ),
    );
    expect(update.response.ok).toBe(true);
    expect(update.body).toMatchObject({
      agent_settings: {
        llm: {
          model: "openai/gpt-5.6-sol",
          api_key: "sk-secret",
          base_url: "https://api.example.test/v1",
          stop: ["DONE"],
          parameters: { reasoning: "high" },
        },
        verification: {
          critic_enabled: true,
          enable_iterative_refinement: false,
        },
        mcp_config: null,
      },
      conversation_settings: {
        max_iterations: 12,
        confirmation_mode: true,
      },
      llm_api_key_is_set: true,
      misc_settings: {
        app_preferences: {
          language: "fr",
          disabled_skills: ["legacy-skill"],
        },
      },
    });
  });

  it("replaces incompatible nested values without leaking their previous shape", async () => {
    const result = await getJson<{
      agent_settings: {
        verification: unknown;
        llm: { model: unknown };
      };
    }>(
      "/api/settings",
      jsonRequest(
        {
          agent_settings_diff: {
            verification: "disabled",
            llm: { model: { family: "custom" } },
          },
        },
        "PATCH",
      ),
    );

    expect(result.body.agent_settings.verification).toBe("disabled");
    expect(result.body.agent_settings.llm.model).toEqual({ family: "custom" });
  });

  it("redacts, encrypts, or exposes persisted settings secrets", async () => {
    await fetch(
      `${BASE_URL}/api/settings`,
      jsonRequest(
        { agent_settings_diff: { llm: { api_key: "sk-abcdefghijk" } } },
        "PATCH",
      ),
    );

    const redacted = await getJson<{
      agent_settings: { llm: { api_key: string } };
    }>("/api/settings");
    expect(redacted.body.agent_settings.llm.api_key).toBe("**********");

    const plaintext = await getJson<{
      agent_settings: { llm: { api_key: string } };
    }>("/api/settings", { headers: { "X-Expose-Secrets": "plaintext" } });
    expect(plaintext.body.agent_settings.llm.api_key).toBe("sk-abcdefghijk");

    const encrypted = await getJson<{
      agent_settings: { llm: { api_key: string } };
    }>("/api/settings", { headers: { "X-Expose-Secrets": "encrypted" } });
    expect(encrypted.body.agent_settings.llm.api_key).toBe(
      "gAAAAA_mock_encrypted_sk-abcde",
    );
  });

  it("does not mark blank or non-string API keys as configured", async () => {
    for (const apiKey of ["", "   ", 42]) {
      resetTestHandlersMockSettings();
      const result = await getJson<{ llm_api_key_is_set: boolean }>(
        "/api/settings",
        jsonRequest(
          { agent_settings_diff: { llm: { api_key: apiKey } } },
          "PATCH",
        ),
      );
      expect(result.body.llm_api_key_is_set).toBe(false);
    }
  });

  it("handles absent LLM settings and independent API-key signals", async () => {
    const withoutLlm = await getJson<{
      agent_settings: { llm: null };
      llm_api_key_is_set: boolean;
    }>(
      "/api/settings",
      jsonRequest({ agent_settings_diff: { llm: null } }, "PATCH"),
    );
    expect(withoutLlm.body).toMatchObject({
      agent_settings: { llm: null },
      llm_api_key_is_set: false,
    });
    const fetchedWithoutLlm = await getJson<{
      agent_settings: { llm: null };
      llm_api_key_is_set: boolean;
    }>("/api/settings");
    expect(fetchedWithoutLlm.body).toMatchObject({
      agent_settings: { llm: null },
      llm_api_key_is_set: false,
    });

    resetTestHandlersMockSettings();
    await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest(
        {
          llm_api_key_set: true,
          agent_settings_diff: { llm: { api_key: null } },
        },
        "POST",
      ),
    );
    const flagOnly = await getJson<{ llm_api_key_is_set: boolean }>(
      "/api/settings",
    );
    expect(flagOnly.body.llm_api_key_is_set).toBe(true);

    resetTestHandlersMockSettings();
    await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest(
        {
          llm_api_key_set: false,
          agent_settings_diff: { llm: { api_key: "nested-key" } },
        },
        "POST",
      ),
    );
    const nestedOnly = await getJson<{ llm_api_key_is_set: boolean }>(
      "/api/settings",
    );
    expect(nestedOnly.body.llm_api_key_is_set).toBe(true);
  });

  it("merges app preferences across incremental updates", async () => {
    const emptyMisc = await getJson<{ misc_settings: unknown }>(
      "/api/settings",
      jsonRequest({ misc_settings_diff: {} }, "PATCH"),
    );
    expect(emptyMisc.body.misc_settings).toEqual({});

    await fetch(
      `${BASE_URL}/api/settings`,
      jsonRequest(
        {
          misc_settings_diff: {
            app_preferences: { language: "de", git_user_name: "Ada" },
          },
        },
        "PATCH",
      ),
    );
    const second = await getJson<{
      misc_settings: { app_preferences: Record<string, unknown> };
    }>(
      "/api/settings",
      jsonRequest(
        {
          misc_settings_diff: {
            app_preferences: {
              language: "es",
              disabled_skills: ["skill-a"],
            },
          },
        },
        "PATCH",
      ),
    );
    expect(second.body.misc_settings.app_preferences).toEqual({
      language: "es",
      git_user_name: "Ada",
      disabled_skills: ["skill-a"],
    });

    const persisted = await getJson<{
      misc_settings: { app_preferences: Record<string, unknown> };
    }>("/api/settings");
    expect(persisted.body.misc_settings.app_preferences).toMatchObject({
      language: "es",
      git_user_name: "Ada",
      git_user_email: null,
      disabled_skills: ["skill-a"],
    });
  });

  it("preserves unrelated persisted misc settings while merging preferences", async () => {
    await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest(
        {
          misc_settings: {
            marker: "keep-me",
            app_preferences: { language: "en" },
          },
        },
        "POST",
      ),
    );

    const updated = await getJson<{
      misc_settings: {
        marker: string;
        app_preferences: Record<string, unknown>;
      };
    }>(
      "/api/settings",
      jsonRequest(
        {
          misc_settings_diff: {
            app_preferences: { git_user_name: "Ada" },
          },
        },
        "PATCH",
      ),
    );

    expect(updated.body.misc_settings).toEqual({
      marker: "keep-me",
      app_preferences: { language: "en", git_user_name: "Ada" },
    });
  });

  it("supports independent conversation-settings updates", async () => {
    const result = await getJson<{
      agent_settings: Record<string, unknown>;
      conversation_settings: Record<string, unknown>;
      misc_settings: { app_preferences: Record<string, unknown> };
    }>(
      "/api/settings",
      jsonRequest(
        { conversation_settings_diff: { max_iterations: 99 } },
        "PATCH",
      ),
    );
    expect(result.body.conversation_settings).toMatchObject({
      confirmation_mode: false,
      max_iterations: 99,
    });
    expect(result.body.agent_settings).toMatchObject({
      llm: { model: "openai/gpt-5.6-sol" },
      verification: { critic_enabled: false },
    });
    expect(result.body.misc_settings).toEqual({ app_preferences: {} });
  });

  it("normalizes a persisted null API-key flag in update responses", async () => {
    await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest({ llm_api_key_set: null }, "POST"),
    );

    const result = await getJson<{ llm_api_key_is_set: boolean }>(
      "/api/settings",
      jsonRequest(
        { conversation_settings_diff: { max_iterations: 3 } },
        "PATCH",
      ),
    );
    expect(result.body.llm_api_key_is_set).toBe(false);
  });

  it("reports successful MCP connectivity with a mock tool", async () => {
    const result = await getJson<{ ok: boolean; tools: string[] }>(
      "/api/mcp/test",
      { method: "POST" },
    );
    expect(result.response.ok).toBe(true);
    expect(result.body).toEqual({ ok: true, tools: ["mock_tool"] });
  });
});

describe("legacy settings persistence", () => {
  it("rejects direct nested settings and identifies offending keys", async () => {
    const both = await getJson<{ error: string; keys: string[] }>(
      "/api/v1/settings",
      jsonRequest({ agent_settings: {}, conversation_settings: {} }, "POST"),
    );
    expect(both.response.status).toBe(422);
    expect(both.body).toEqual({
      error: "Use *_diff nested settings payloads",
      keys: ["agent_settings", "conversation_settings"],
    });

    const one = await getJson<{ keys: string[] }>(
      "/api/v1/settings",
      jsonRequest({ agent_settings: {} }, "POST"),
    );
    expect(one.response.status).toBe(422);
    expect(one.body.keys).toEqual(["agent_settings"]);

    const conversationOnly = await getJson<{ error: string; keys: string[] }>(
      "/api/v1/settings",
      jsonRequest({ conversation_settings: {} }, "POST"),
    );
    expect(conversationOnly.response.status).toBe(422);
    expect(conversationOnly.body).toEqual({
      error: "Use *_diff nested settings payloads",
      keys: ["conversation_settings"],
    });
  });

  it("accepts nested diffs and independent top-level settings", async () => {
    const response = await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest(
        {
          agent_settings_diff: {
            llm: { model: "openai/gpt-5.5" },
            verification: { critic_enabled: true },
          },
          conversation_settings_diff: {
            max_iterations: 7,
          },
          agent_settings_schema: { ignored: true },
          conversation_settings_schema: { ignored: true },
          language: "ja",
          llm_api_key_set: true,
        },
        "POST",
      ),
    );
    expect(response.status).toBe(200);

    const persisted = await getJson<{
      agent_settings: Record<string, unknown>;
      conversation_settings: Record<string, unknown>;
      agent_settings_schema: { model_name: string };
      conversation_settings_schema: { model_name: string };
      language: string;
      llm_api_key_set: boolean;
    }>("/api/v1/settings");
    expect(persisted.body).toMatchObject({
      agent_settings: {
        llm: { model: "openai/gpt-5.5" },
        verification: {
          critic_enabled: true,
          enable_iterative_refinement: false,
        },
      },
      conversation_settings: {
        confirmation_mode: false,
        max_iterations: 7,
      },
      language: "ja",
      llm_api_key_set: true,
    });
    expect(persisted.body).not.toHaveProperty("agent_settings_diff");
    expect(persisted.body).not.toHaveProperty("conversation_settings_diff");
    expect(persisted.body.agent_settings_schema).toMatchObject({
      model_name: "AgentSettings",
    });
    expect(persisted.body.conversation_settings_schema).toMatchObject({
      model_name: "ConversationSettings",
    });
  });

  it("accepts an empty object but rejects a null body", async () => {
    const empty = await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest({}, "POST"),
    );
    expect(empty.status).toBe(200);

    const nullBody = await fetch(
      `${BASE_URL}/api/v1/settings`,
      jsonRequest(null, "POST"),
    );
    expect(nullBody.status).toBe(400);
    await expect(nullBody.json()).resolves.toBeNull();
  });
});
