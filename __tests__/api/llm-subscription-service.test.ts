import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import LLMSubscriptionService from "#/api/llm-subscription-service";
import {
  getActiveSelection,
  getRegisteredBackends,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH,
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
  OPENAI_SUBSCRIPTION_MODELS_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
} from "#/constants/llm-subscription";
import { server } from "#/mocks/node";
import { resetTestHandlersMockSettings } from "#/mocks/settings-handlers";

describe("LLMSubscriptionService", () => {
  beforeEach(() => {
    resetTestHandlersMockSettings();
  });

  it("fetches OpenAI subscription models from the agent-server endpoint", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, ({ request }) => {
        expect(request.headers.get("Accept")).toBeNull();
        expect(request.headers.get("Content-Type")).toBe("application/json");
        expect(request.headers.get("X-Session-API-Key")).toBe(
          "test-session-key",
        );
        return HttpResponse.json({
          models: ["gpt-5.2", "gpt-5.3-codex"],
        });
      }),
    );

    await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual([
      "gpt-5.2",
      "gpt-5.3-codex",
    ]);
  });

  it("normalizes a top-level model list and excludes non-string entries", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, () =>
        HttpResponse.json(["gpt-direct", 42, null]),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual([
      "gpt-direct",
    ]);
  });

  it("filters non-string entries from a nested model list", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, () =>
        HttpResponse.json({ models: ["gpt-nested", 42, null] }),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual([
      "gpt-nested",
    ]);
  });

  it.each([
    ["null", null],
    ["a primitive string", "gpt-not-a-list"],
    ["a primitive number", 42],
    ["an object whose models field is not an array", { models: "gpt" }],
  ])("returns no models when the payload is %s", async (_label, payload) => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, () =>
        HttpResponse.json(payload),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual([]);
  });

  it("omits session authentication when the local backend has no API key", async () => {
    const originalBackends = getRegisteredBackends();
    const originalSelection = getActiveSelection();
    const backendWithoutKey = {
      id: "local-without-key",
      name: "Local without key",
      host: "http://localhost",
      apiKey: "",
      kind: "local" as const,
    };

    try {
      setRegisteredBackends([backendWithoutKey]);
      setActiveSelection({ backendId: backendWithoutKey.id });
      server.use(
        http.get(`*${OPENAI_SUBSCRIPTION_MODELS_PATH}`, ({ request }) => {
          expect(request.headers.has("X-Session-API-Key")).toBe(false);
          return HttpResponse.json([]);
        }),
      );

      await expect(LLMSubscriptionService.getOpenAIModels()).resolves.toEqual(
        [],
      );
    } finally {
      setRegisteredBackends(originalBackends);
      setActiveSelection(originalSelection);
    }
  });

  it("normalizes OpenAI subscription status from MSW handlers", async () => {
    await expect(LLMSubscriptionService.getOpenAIStatus()).resolves.toEqual({
      vendor: "openai",
      connected: false,
      accountEmail: null,
      expiresAt: null,
    });
  });

  it.each([
    {
      label: "primary snake-case fields",
      payload: {
        connected: true,
        account_email: " primary@example.com ",
        expires_at: " 2030-01-01T00:00:00Z ",
      },
      expected: {
        connected: true,
        accountEmail: "primary@example.com",
        expiresAt: "2030-01-01T00:00:00Z",
      },
    },
    {
      label: "secondary aliases",
      payload: {
        authenticated: true,
        email: " secondary@example.com ",
        expiresAt: " 2031-01-01T00:00:00Z ",
      },
      expected: {
        connected: true,
        accountEmail: "secondary@example.com",
        expiresAt: "2031-01-01T00:00:00Z",
      },
    },
    {
      label: "tertiary boolean and account aliases",
      payload: {
        is_connected: true,
        account: " tertiary@example.com ",
        expires_at: 101,
      },
      expected: {
        connected: true,
        accountEmail: "tertiary@example.com",
        expiresAt: 101,
      },
    },
    {
      label: "numeric camel-case expiry",
      payload: { connected: false, expiresAt: 202 },
      expected: {
        connected: false,
        accountEmail: null,
        expiresAt: 202,
      },
    },
    {
      label: "blank and invalid primary aliases",
      payload: {
        connected: "not-a-boolean",
        authenticated: false,
        account_email: "   ",
        email: " fallback@example.com ",
        expires_at: "   ",
        expiresAt: " fallback-expiry ",
      },
      expected: {
        connected: false,
        accountEmail: "fallback@example.com",
        expiresAt: "fallback-expiry",
      },
    },
  ])("normalizes $label", async ({ payload, expected }) => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, () =>
        HttpResponse.json(payload),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIStatus()).resolves.toEqual({
      vendor: "openai",
      ...expected,
    });
  });

  it("skips non-finite numeric status values in favor of later aliases", async () => {
    server.use(
      http.get(
        `*${OPENAI_SUBSCRIPTION_STATUS_PATH}`,
        () =>
          new HttpResponse(
            '{"connected":true,"expires_at":1e400,"expiresAt":303}',
            { headers: { "Content-Type": "application/json" } },
          ),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIStatus()).resolves.toEqual({
      vendor: "openai",
      connected: true,
      accountEmail: null,
      expiresAt: 303,
    });
  });

  it("defaults to disconnected when status flags are not booleans", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, () =>
        HttpResponse.json({
          connected: "yes",
          authenticated: 1,
          is_connected: null,
        }),
      ),
    );

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

  it.each([
    {
      label: "snake-case string fields",
      payload: {
        device_code: " device-snake ",
        user_code: " user-snake ",
        verification_uri: " https://verify.example/snake ",
        verification_uri_complete:
          " https://verify.example/snake?code=user-snake ",
        expires_at: " 2030-01-01T00:00:00Z ",
        interval: 1,
      },
      expected: {
        deviceCode: "device-snake",
        userCode: "user-snake",
        verificationUri: "https://verify.example/snake",
        verificationUriComplete: "https://verify.example/snake?code=user-snake",
        expiresAt: "2030-01-01T00:00:00Z",
        intervalSeconds: 1,
      },
    },
    {
      label: "camel-case fields after blank or invalid primary aliases",
      payload: {
        device_code: "   ",
        deviceCode: " device-camel ",
        user_code: 42,
        userCode: " user-camel ",
        verification_uri: "   ",
        verificationUri: " https://verify.example/camel ",
        verificationUriComplete:
          " https://verify.example/camel?code=user-camel ",
        expiresAt: " 2031-01-01T00:00:00Z ",
        interval_seconds: 2,
      },
      expected: {
        deviceCode: "device-camel",
        userCode: "user-camel",
        verificationUri: "https://verify.example/camel",
        verificationUriComplete: "https://verify.example/camel?code=user-camel",
        expiresAt: "2031-01-01T00:00:00Z",
        intervalSeconds: 2,
      },
    },
    {
      label: "snake-case URL aliases and numeric snake-case expiry",
      payload: {
        device_code: "device-url-snake",
        user_code: "user-url-snake",
        verification_url: "https://verify.example/url-snake",
        verification_url_complete:
          "https://verify.example/url-snake?code=user-url-snake",
        expires_at: 303,
        intervalSeconds: 3,
      },
      expected: {
        deviceCode: "device-url-snake",
        userCode: "user-url-snake",
        verificationUri: "https://verify.example/url-snake",
        verificationUriComplete:
          "https://verify.example/url-snake?code=user-url-snake",
        expiresAt: 303,
        intervalSeconds: 3,
      },
    },
    {
      label: "camel-case URL aliases and numeric camel-case expiry",
      payload: {
        device_code: "device-url-camel",
        user_code: "user-url-camel",
        verificationUrl: "https://verify.example/url-camel",
        verificationUrlComplete:
          "https://verify.example/url-camel?code=user-url-camel",
        expiresAt: 404,
      },
      expected: {
        deviceCode: "device-url-camel",
        userCode: "user-url-camel",
        verificationUri: "https://verify.example/url-camel",
        verificationUriComplete:
          "https://verify.example/url-camel?code=user-url-camel",
        expiresAt: 404,
        intervalSeconds: null,
      },
    },
    {
      label: "relative snake-case expiry",
      payload: {
        device_code: "device-expires-in",
        user_code: "user-expires-in",
        verification_uri: "https://verify.example/expires-in",
        expires_in: 505,
      },
      expected: {
        deviceCode: "device-expires-in",
        userCode: "user-expires-in",
        verificationUri: "https://verify.example/expires-in",
        verificationUriComplete: null,
        expiresAt: 505,
        intervalSeconds: null,
      },
    },
    {
      label: "relative camel-case expiry",
      payload: {
        device_code: "device-expires-camel",
        user_code: "user-expires-camel",
        verification_uri: "https://verify.example/expires-camel",
        expiresIn: 606,
      },
      expected: {
        deviceCode: "device-expires-camel",
        userCode: "user-expires-camel",
        verificationUri: "https://verify.example/expires-camel",
        verificationUriComplete: null,
        expiresAt: 606,
        intervalSeconds: null,
      },
    },
  ])("normalizes device challenge $label", async ({ payload, expected }) => {
    server.use(
      http.post(`*${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`, () =>
        HttpResponse.json(payload),
      ),
    );

    await expect(
      LLMSubscriptionService.startOpenAIDeviceLogin(),
    ).resolves.toEqual(expected);
  });

  it("skips non-finite numeric device values in favor of later aliases", async () => {
    server.use(
      http.post(
        `*${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`,
        () =>
          new HttpResponse(
            '{"device_code":"device","user_code":"user","verification_uri":"https://verify.example","expires_at":1e400,"expiresAt":707,"interval":1e400,"interval_seconds":7}',
            { headers: { "Content-Type": "application/json" } },
          ),
      ),
    );

    await expect(
      LLMSubscriptionService.startOpenAIDeviceLogin(),
    ).resolves.toEqual({
      deviceCode: "device",
      userCode: "user",
      verificationUri: "https://verify.example",
      verificationUriComplete: null,
      expiresAt: 707,
      intervalSeconds: 7,
    });
  });

  it("sends the exact polling body and JSON request headers", async () => {
    server.use(
      http.post(
        `*${OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH}`,
        async ({ request }) => {
          expect(request.headers.get("Accept")).toBeNull();
          expect(request.headers.get("Content-Type")).toBe("application/json");
          expect(request.headers.get("X-Session-API-Key")).toBe(
            "test-session-key",
          );
          await expect(request.text()).resolves.toBe(
            '{"device_code":"exact-device-code"}',
          );
          return HttpResponse.json({ connected: true });
        },
      ),
    );

    await expect(
      LLMSubscriptionService.pollOpenAIDeviceLogin("exact-device-code"),
    ).resolves.toEqual({
      vendor: "openai",
      connected: true,
      accountEmail: null,
      expiresAt: null,
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

  it.each([
    {
      label: "device code",
      payload: {
        device_code: "   ",
        user_code: "MOCK-CODE",
        verification_uri: "https://auth.openai.com/activate",
      },
    },
    {
      label: "user code",
      payload: {
        device_code: "mock-device-code",
        user_code: "   ",
        verification_uri: "https://auth.openai.com/activate",
      },
    },
    {
      label: "verification URI",
      payload: {
        device_code: "mock-device-code",
        user_code: "MOCK-CODE",
        verification_uri: "   ",
      },
    },
  ])("rejects a challenge without a valid $label", async ({ payload }) => {
    server.use(
      http.post(`*${OPENAI_SUBSCRIPTION_DEVICE_START_PATH}`, () =>
        HttpResponse.json(payload),
      ),
    );

    await expect(
      LLMSubscriptionService.startOpenAIDeviceLogin(),
    ).rejects.toThrowError(
      new Error("Subscription device login response is incomplete"),
    );
  });

  it("surfaces agent-server errors", async () => {
    server.use(
      http.get(`*${OPENAI_SUBSCRIPTION_STATUS_PATH}`, () =>
        HttpResponse.json({ detail: "unauthorized" }, { status: 401 }),
      ),
    );

    await expect(LLMSubscriptionService.getOpenAIStatus()).rejects.toThrowError(
      "HTTP request failed (401 Unauthorized)",
    );
  });
});
