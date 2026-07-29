import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import {
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
} from "#/constants/llm-subscription";
import { server } from "#/mocks/node";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

describe("LLMSubscriptionService", () => {
  beforeEach(() => {
    resetTestHandlersMockSettings();
  });

  it("fetches OpenAI subscription models from the agent-server endpoint", async () => {
    await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual([
      "gpt-5.2",
      "gpt-5.3-codex",
    ]);
  });

  it("normalizes OpenAI subscription status from MSW handlers", async () => {
    await expect(LLMSubscriptionService.getOpenAIStatus()).resolves.toEqual({
      vendor: "openai",
      connected: false,
      accountEmail: null,
      expiresAt: null,
    });
  });

  it("normalizes device login challenge responses", async () => {
    await expect(
      LLMSubscriptionService.startOpenAIDeviceLogin(),
    ).resolves.toEqual({
      deviceCode: "mock-device-code",
      userCode: "MOCK-CODE",
      verificationUri: "https://auth.openai.com/activate",
      verificationUriComplete:
        "https://auth.openai.com/activate?user_code=MOCK-CODE",
      expiresAt: 900,
      intervalSeconds: 1,
    });
  });

  it("posts the device code when polling login", async () => {
    await expect(
      LLMSubscriptionService.pollOpenAIDeviceLogin("mock-device-code"),
    ).resolves.toMatchObject({ connected: true });

    await expect(
      LLMSubscriptionService.getOpenAIStatus(),
    ).resolves.toMatchObject({
      connected: true,
      accountEmail: "mock-chatgpt@example.com",
    });
  });

  it("calls the logout endpoint", async () => {
    await LLMSubscriptionService.pollOpenAIDeviceLogin("mock-device-code");

    await expect(LLMSubscriptionService.logoutOpenAI()).resolves.toMatchObject({
      connected: false,
    });
    await expect(
      LLMSubscriptionService.getOpenAIStatus(),
    ).resolves.toMatchObject({ connected: false });
  });

  it("rejects incomplete device challenges with blank required fields", async () => {
    server.use(
      http.post(`*${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`, () =>
        HttpResponse.json({
          device_code: "   ",
          user_code: "MOCK-CODE",
          verification_uri: "https://auth.openai.com/activate",
        }),
      ),
    );

    await expect(
      LLMSubscriptionService.startOpenAIDeviceLogin(),
    ).rejects.toThrow("Subscription device login response is incomplete");
  });

  it("surfaces agent-server errors", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, () =>
        HttpResponse.json({ detail: "unauthorized" }, { status: 401 }),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIStatus()).rejects.toThrow();
  });
});
