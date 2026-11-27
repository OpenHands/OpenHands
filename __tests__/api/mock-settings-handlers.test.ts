import { beforeEach, describe, expect, it } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

beforeEach(() => {
  resetTestHandlersMockSettings();
});

describe("mock settings handlers", () => {
  it("returns the agent settings schema on the paths used by the UI", async () => {
    const schema = await SettingsService.getSettingsSchema();

    expect(schema.sections.some((section) => section.key === "llm")).toBe(true);
  });

  it("returns the conversation settings schema on the paths used by the UI", async () => {
    const schema = await SettingsService.getConversationSettingsSchema();

    expect(
      schema.sections.some((section) => section.key === "verification"),
    ).toBe(true);
  });

  it("supports the profile endpoints used by local LLM settings", async () => {
    const saveResponse = await fetch(
      "http://localhost:3000/api/profiles/mock-profile",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm: {
            model: "openai/gpt-4o",
            base_url: "https://api.openai.com/v1",
            api_key: "sk-test",
          },
          include_secrets: true,
        }),
      },
    );

    expect(saveResponse.status).toBe(201);

    const listResponse = await fetch("http://localhost:3000/api/profiles");
    const list = (await listResponse.json()) as {
      profiles: { name: string; model: string; api_key_set: boolean }[];
      active_profile: string | null;
    };

    expect(list.profiles).toEqual([
      expect.objectContaining({
        name: "mock-profile",
        model: "openai/gpt-4o",
        api_key_set: true,
      }),
    ]);
    expect(list.active_profile).toBeNull();

    const detailResponse = await fetch(
      "http://localhost:3000/api/profiles/mock-profile",
      { headers: { "X-Expose-Secrets": "encrypted" } },
    );
    const detail = (await detailResponse.json()) as {
      config: { api_key?: string };
    };
    expect(detail.config.api_key).toMatch(/^gAAAAA_mock_encrypted_/);

    const activateResponse = await fetch(
      "http://localhost:3000/api/profiles/mock-profile/activate",
      { method: "POST" },
    );
    expect(activateResponse.status).toBe(200);

    const settingsResponse = await fetch("http://localhost:3000/api/settings");
    const settings = (await settingsResponse.json()) as {
      agent_settings: { llm?: { model?: string } };
      llm_api_key_is_set: boolean;
    };
    expect(settings.agent_settings.llm?.model).toBe("openai/gpt-4o");
    expect(settings.llm_api_key_is_set).toBe(true);

    const renameResponse = await fetch(
      "http://localhost:3000/api/profiles/mock-profile/rename",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: "renamed-profile" }),
      },
    );
    expect(renameResponse.status).toBe(200);

    const deleteResponse = await fetch(
      "http://localhost:3000/api/profiles/renamed-profile",
      { method: "DELETE" },
    );
    expect(deleteResponse.status).toBe(200);

    const finalList = (await (
      await fetch("http://localhost:3000/api/profiles")
    ).json()) as { profiles: unknown[]; active_profile: string | null };
    expect(finalList.profiles).toEqual([]);
    expect(finalList.active_profile).toBeNull();
  });
});
