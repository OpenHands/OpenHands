import { beforeEach, describe, expect, it } from "vitest";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

const BASE_URL = "http://localhost:3000";

const jsonRequest = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const saveProfile = (name: string, llm: Record<string, unknown>) =>
  fetch(`${BASE_URL}/api/profiles/${encodeURIComponent(name)}`, {
    ...jsonRequest({ llm }),
  });

const getProfile = (name: string, exposeSecrets?: string) =>
  fetch(`${BASE_URL}/api/profiles/${encodeURIComponent(name)}`, {
    headers: exposeSecrets ? { "X-Expose-Secrets": exposeSecrets } : {},
  });

beforeEach(() => {
  resetTestHandlersMockSettings();
});

describe("mock LLM profile lifecycle", () => {
  it("rejects empty requests and profiles without a usable model", async () => {
    const empty = await fetch(
      `${BASE_URL}/api/profiles/empty`,
      jsonRequest(null),
    );
    expect(empty.status).toBe(400);
    await expect(empty.json()).resolves.toEqual({ detail: "Empty body" });

    const missingLlm = await fetch(
      `${BASE_URL}/api/profiles/invalid`,
      jsonRequest({}),
    );
    expect(missingLlm.status).toBe(400);

    for (const llm of [{}, { model: 42 }, { model: "   " }]) {
      const response = await saveProfile("invalid", llm);
      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual({
        detail: "Profile requires llm.model",
      });
    }
  });

  it("lists profiles with normalized optional summary fields", async () => {
    const saved = await saveProfile("model only", {
      model: "openai/gpt-4o",
      base_url: "https://api.openai.com/v1",
    });
    expect(saved.status).toBe(201);
    await expect(saved.json()).resolves.toEqual({
      name: "model only",
      message: "Profile 'model only' saved",
    });
    expect(
      (
        await saveProfile("non-string fields", {
          model: "anthropic/claude-3.5",
          base_url: 42,
          api_key: "   ",
        })
      ).status,
    ).toBe(201);

    const response = await fetch(`${BASE_URL}/api/profiles`);
    await expect(response.json()).resolves.toEqual({
      profiles: [
        {
          name: "model only",
          model: "openai/gpt-4o",
          base_url: "https://api.openai.com/v1",
          api_key_set: false,
        },
        {
          name: "non-string fields",
          model: "anthropic/claude-3.5",
          base_url: null,
          api_key_set: false,
        },
      ],
      active_profile: null,
    });
  });

  it("returns not found for an unknown profile", async () => {
    const response = await getProfile("missing profile");
    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      detail: "Profile 'missing profile' not found",
    });
  });

  it("redacts, encrypts, or exposes a configured API key", async () => {
    await saveProfile("secret profile", {
      model: "openai/gpt-4o",
      base_url: "https://api.openai.com/v1",
      api_key: "sk-plaintext",
    });

    const redacted = await getProfile("secret profile");
    await expect(redacted.json()).resolves.toMatchObject({
      name: "secret profile",
      config: { api_key: null },
      api_key_set: true,
    });

    const plaintext = await getProfile("secret profile", "plaintext");
    await expect(plaintext.json()).resolves.toMatchObject({
      config: { api_key: "sk-plaintext" },
    });

    const encrypted = await getProfile("secret profile", "encrypted");
    await expect(encrypted.json()).resolves.toMatchObject({
      config: { api_key: "gAAAAA_mock_encrypted_secret profile" },
    });
  });

  it("leaves a non-configured API key unchanged in profile details", async () => {
    await saveProfile("blank key", {
      model: "openai/gpt-4o",
      api_key: "   ",
    });

    const response = await getProfile("blank key");
    await expect(response.json()).resolves.toMatchObject({
      config: { api_key: "   " },
      api_key_set: false,
    });
  });

  it("rejects activation of an unknown profile", async () => {
    const response = await fetch(`${BASE_URL}/api/profiles/missing/activate`, {
      method: "POST",
    });
    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      detail: "Profile 'missing' not found",
    });
  });

  it("applies an activated profile and reapplies later saves", async () => {
    await saveProfile("active", {
      model: "openai/gpt-4o",
      api_key: "first-key",
    });
    const activation = await fetch(`${BASE_URL}/api/profiles/active/activate`, {
      method: "POST",
    });
    expect(activation.status).toBe(200);
    await expect(activation.json()).resolves.toEqual({
      name: "active",
      message: "Profile 'active' activated and applied to current settings",
      llm_applied: true,
    });

    await saveProfile("inactive", { model: "openhands/minimax-m2.7" });
    const unchangedSettings = await fetch(`${BASE_URL}/api/settings`);
    await expect(unchangedSettings.json()).resolves.toMatchObject({
      agent_settings: { llm: { model: "openai/gpt-4o" } },
      llm_api_key_is_set: true,
    });

    await saveProfile("active", {
      model: "anthropic/claude-opus-4-8",
    });

    const settings = await fetch(`${BASE_URL}/api/settings`);
    await expect(settings.json()).resolves.toMatchObject({
      agent_settings: {
        llm: { model: "anthropic/claude-opus-4-8" },
      },
      llm_api_key_is_set: false,
    });
  });

  it("validates rename requests and prevents collisions", async () => {
    const missing = await fetch(
      `${BASE_URL}/api/profiles/missing/rename`,
      jsonRequest({ new_name: "new" }),
    );
    expect(missing.status).toBe(404);
    await expect(missing.json()).resolves.toEqual({
      detail: "Profile 'missing' not found",
    });

    await saveProfile("first", { model: "openai/gpt-4o" });
    await saveProfile("second", { model: "openai/gpt-5.5" });

    for (const body of [null, {}, { new_name: "   " }]) {
      const invalid = await fetch(
        `${BASE_URL}/api/profiles/first/rename`,
        jsonRequest(body),
      );
      expect(invalid.status).toBe(400);
      await expect(invalid.json()).resolves.toEqual({
        detail: "new_name is required",
      });
    }

    const collision = await fetch(
      `${BASE_URL}/api/profiles/first/rename`,
      jsonRequest({ new_name: "second" }),
    );
    expect(collision.status).toBe(409);
    await expect(collision.json()).resolves.toEqual({
      detail: "Profile 'second' already exists",
    });
  });

  it("supports no-op and inactive profile renames", async () => {
    await saveProfile("unchanged", { model: "openai/gpt-4o" });

    const sameName = await fetch(
      `${BASE_URL}/api/profiles/unchanged/rename`,
      jsonRequest({ new_name: " unchanged " }),
    );
    expect(sameName.status).toBe(200);
    await expect(sameName.json()).resolves.toEqual({
      name: "unchanged",
      message: "Profile 'unchanged' renamed to 'unchanged'",
    });

    const renamed = await fetch(
      `${BASE_URL}/api/profiles/unchanged/rename`,
      jsonRequest({ new_name: "renamed" }),
    );
    expect(renamed.status).toBe(200);
    await expect(renamed.json()).resolves.toEqual({
      name: "renamed",
      message: "Profile 'unchanged' renamed to 'renamed'",
    });

    const list = await fetch(`${BASE_URL}/api/profiles`);
    await expect(list.json()).resolves.toMatchObject({ active_profile: null });
  });

  it("keeps an active profile active when it is renamed", async () => {
    await saveProfile("active", {
      model: "openai/gpt-4o",
      api_key: "sk-test",
    });
    await fetch(`${BASE_URL}/api/profiles/active/activate`, { method: "POST" });

    const response = await fetch(
      `${BASE_URL}/api/profiles/active/rename`,
      jsonRequest({ new_name: "renamed active" }),
    );
    expect(response.status).toBe(200);

    const list = await fetch(`${BASE_URL}/api/profiles`);
    await expect(list.json()).resolves.toMatchObject({
      profiles: [expect.objectContaining({ name: "renamed active" })],
      active_profile: "renamed active",
    });
    const settings = await fetch(`${BASE_URL}/api/settings`);
    await expect(settings.json()).resolves.toMatchObject({
      agent_settings: { llm: { model: "openai/gpt-4o" } },
      llm_api_key_is_set: true,
    });
  });

  it("deletes inactive and active profiles", async () => {
    await saveProfile("inactive", { model: "openai/gpt-4o" });
    await saveProfile("active", { model: "openai/gpt-5.5" });
    await fetch(`${BASE_URL}/api/profiles/active/activate`, { method: "POST" });

    const inactiveDelete = await fetch(`${BASE_URL}/api/profiles/inactive`, {
      method: "DELETE",
    });
    expect(inactiveDelete.status).toBe(200);
    await expect(inactiveDelete.json()).resolves.toEqual({
      name: "inactive",
      message: "Profile 'inactive' deleted",
    });

    const afterInactiveDelete = await fetch(`${BASE_URL}/api/profiles`);
    await expect(afterInactiveDelete.json()).resolves.toMatchObject({
      active_profile: "active",
    });

    const activeDelete = await fetch(`${BASE_URL}/api/profiles/active`, {
      method: "DELETE",
    });
    expect(activeDelete.status).toBe(200);
    await expect(activeDelete.json()).resolves.toEqual({
      name: "active",
      message: "Profile 'active' deleted",
    });

    const list = await fetch(`${BASE_URL}/api/profiles`);
    await expect(list.json()).resolves.toEqual({
      profiles: [],
      active_profile: null,
    });
  });
});
