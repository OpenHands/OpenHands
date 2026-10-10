import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import {
  assertSubscriptionAuthReady,
  SUBSCRIPTION_LOGIN_REQUIRED_ERROR,
} from "#/api/agent-server-adapter";
import {
  OPENAI_SUBSCRIPTION_MODELS_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
} from "#/constants/llm-subscription";
import { server } from "#/mocks/node";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

function subscriptionLlm(model: unknown) {
  return { llm: { auth_type: "subscription", model } };
}

function mockConnectedStatus() {
  server.use(
    http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, () =>
      HttpResponse.json({ connected: true }),
    ),
  );
}

function mockAdvertisedModels(models: unknown) {
  server.use(
    http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, () =>
      HttpResponse.json({ models }),
    ),
  );
}

describe("assertSubscriptionAuthReady", () => {
  beforeEach(() => {
    resetTestHandlersMockSettings();
  });

  it("ignores configs that do not use subscription auth", async () => {
    await expect(
      assertSubscriptionAuthReady({ llm: { model: "openai/gpt-5.5" } }),
    ).resolves.toBeUndefined();
  });

  it("rejects disconnected subscription profiles before starting", async () => {
    await expect(
      assertSubscriptionAuthReady(subscriptionLlm("gpt-5.2")),
    ).rejects.toThrow(SUBSCRIPTION_LOGIN_REQUIRED_ERROR);
  });

  it("allows a connected subscription whose model is advertised", async () => {
    mockConnectedStatus();
    await expect(
      assertSubscriptionAuthReady(subscriptionLlm("gpt-5.2")),
    ).resolves.toBeUndefined();
  });

  it("accepts a provider-prefixed model matching an advertised bare id", async () => {
    mockConnectedStatus();
    await expect(
      assertSubscriptionAuthReady(subscriptionLlm("openai/gpt-5.2")),
    ).resolves.toBeUndefined();
  });

  it("rejects a connected subscription whose model is not served", async () => {
    mockConnectedStatus();
    const error = await assertSubscriptionAuthReady(
      subscriptionLlm("openai/gpt-5.6-terra"),
    ).catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain("openai/gpt-5.6-terra");
    expect((error as Error).message).toContain("not available");
  });

  it("fails open when the server advertises no models", async () => {
    mockConnectedStatus();
    mockAdvertisedModels([]);
    await expect(
      assertSubscriptionAuthReady(subscriptionLlm("openai/gpt-5.6-terra")),
    ).resolves.toBeUndefined();
  });

  it("fails open when the profile carries no model", async () => {
    mockConnectedStatus();
    await expect(
      assertSubscriptionAuthReady(subscriptionLlm("")),
    ).resolves.toBeUndefined();
  });
});
