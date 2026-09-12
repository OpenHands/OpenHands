import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import {
  startDeviceFlow,
  pollForToken,
  isOpenHandsCloudHost,
  DeviceFlowError,
} from "../../src/api/device-flow-client";
import {
  AGENT_CANVAS_CLIENT_HEADERS,
  OPENHANDS_CLIENT_HEADER,
  OPENHANDS_CLIENT_VERSION_HEADER,
} from "../../src/api/client-source";

const TEST_HOST_URL = "https://app.all-hands.dev";
const AUTHORIZE_URL = `${TEST_HOST_URL}/oauth/device/authorize`;
const TOKEN_URL = `${TEST_HOST_URL}/oauth/device/token`;

// The agent-canvas wrapper's only job is to forward requests to the SDK while
// attaching the coarse observability headers. Every fetch-backed assertion goes
// through MSW so we can inspect the actual outgoing request (URL, body, and the
// forwarded headers) rather than mocking `global.fetch`.
function expectClientHeaders(headers: Headers) {
  expect(headers.get(OPENHANDS_CLIENT_HEADER)).toBe(
    AGENT_CANVAS_CLIENT_HEADERS[OPENHANDS_CLIENT_HEADER],
  );
  expect(headers.get(OPENHANDS_CLIENT_VERSION_HEADER)).toBe(
    AGENT_CANVAS_CLIENT_HEADERS[OPENHANDS_CLIENT_VERSION_HEADER],
  );
}

describe("device-flow-client", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe("isOpenHandsCloudHost", () => {
    it("returns true for all-hands.dev domains", () => {
      expect(isOpenHandsCloudHost("https://app.all-hands.dev")).toBe(true);
      expect(isOpenHandsCloudHost("https://staging.all-hands.dev")).toBe(true);
      expect(isOpenHandsCloudHost("app.all-hands.dev")).toBe(true);
      expect(isOpenHandsCloudHost("ALL-HANDS.DEV")).toBe(true);
      expect(isOpenHandsCloudHost("all-hands.dev")).toBe(true);
    });

    it("accepts HTTP cloud URLs surrounded by whitespace", () => {
      expect(isOpenHandsCloudHost("  http://app.all-hands.dev  ")).toBe(true);
    });

    it("returns true for openhands.dev domains", () => {
      expect(isOpenHandsCloudHost("https://app.openhands.dev")).toBe(true);
      expect(isOpenHandsCloudHost("openhands.dev")).toBe(true);
    });

    it("returns false for other domains", () => {
      expect(isOpenHandsCloudHost("https://localhost:8000")).toBe(false);
      expect(isOpenHandsCloudHost("http://127.0.0.1")).toBe(false);
      expect(isOpenHandsCloudHost("https://example.com")).toBe(false);
      expect(isOpenHandsCloudHost("https://my-openhands-server.com")).toBe(
        false,
      );
    });

    it("prevents substring matching attacks", () => {
      // These should NOT be treated as trusted hosts
      expect(isOpenHandsCloudHost("https://all-hands.dev.evil.com")).toBe(
        false,
      );
      expect(isOpenHandsCloudHost("https://malicious-all-hands.dev")).toBe(
        false,
      );
      expect(isOpenHandsCloudHost("https://evil.com/all-hands.dev")).toBe(
        false,
      );
      expect(isOpenHandsCloudHost("prefixhttps://app.all-hands.dev")).toBe(
        false,
      );
    });

    it("returns false for invalid URLs", () => {
      expect(isOpenHandsCloudHost("")).toBe(false);
      expect(isOpenHandsCloudHost("not-a-url")).toBe(false);
    });
  });

  describe("startDeviceFlow", () => {
    it("returns device authorization response and forwards client headers", async () => {
      const mockResponse = {
        device_code: "device123",
        user_code: "USER-1234",
        verification_uri: `${TEST_HOST_URL}/device`,
        verification_uri_complete: `${TEST_HOST_URL}/device?user_code=USER-1234`,
        expires_in: 600,
        interval: 5,
      };

      let requestUrl: string | undefined;
      let requestBody: string | undefined;
      let requestHeaders: Headers | undefined;
      server.use(
        http.post(AUTHORIZE_URL, async ({ request }) => {
          requestUrl = request.url;
          requestBody = await request.text();
          requestHeaders = request.headers;
          return HttpResponse.json(mockResponse);
        }),
      );

      const result = await startDeviceFlow(TEST_HOST_URL);

      expect(result).toEqual(mockResponse);
      expect(requestUrl).toBe(AUTHORIZE_URL);
      expect(requestBody).toBe("{}");
      expect(requestHeaders?.get("content-type")).toBe("application/json");
      // The wrapper must attach the agent-canvas observability headers.
      expectClientHeaders(requestHeaders as Headers);
    });

    it("builds optional authorization values from the required response fields", async () => {
      server.use(
        http.post(AUTHORIZE_URL, () =>
          HttpResponse.json({
            device_code: "device123",
            user_code: "USER 12/+",
            verification_uri: `${TEST_HOST_URL}/device`,
          }),
        ),
      );

      await expect(startDeviceFlow(TEST_HOST_URL)).resolves.toEqual({
        device_code: "device123",
        user_code: "USER 12/+",
        verification_uri: `${TEST_HOST_URL}/device`,
        verification_uri_complete: `${TEST_HOST_URL}/device?user_code=USER%2012%2F%2B`,
        expires_in: 600,
        interval: 5,
      });
    });

    it("normalizes host URL by removing trailing slashes", async () => {
      let requestUrl: string | undefined;
      server.use(
        http.post(AUTHORIZE_URL, ({ request }) => {
          requestUrl = request.url;
          return HttpResponse.json({
            device_code: "dc",
            user_code: "uc",
            verification_uri: "v",
            verification_uri_complete: "vc",
            expires_in: 600,
            interval: 5,
          });
        }),
      );

      await startDeviceFlow(`${TEST_HOST_URL}///`);

      // Verify the direct request targets the normalized host.
      expect(requestUrl).toBe(AUTHORIZE_URL);
    });

    it("throws DeviceFlowError on HTTP error", async () => {
      server.use(
        http.post(
          AUTHORIZE_URL,
          () => new HttpResponse("Internal Server Error", { status: 500 }),
        ),
      );

      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Failed to start device flow: Server returned 500",
      });
    });

    it.each([
      {
        field: "device_code",
        response: { user_code: "uc", verification_uri: "v" },
      },
      {
        field: "user_code",
        response: { device_code: "dc", verification_uri: "v" },
      },
      {
        field: "verification_uri",
        response: { device_code: "dc", user_code: "uc" },
      },
    ])(
      "throws DeviceFlowError when $field is missing",
      async ({ response }) => {
        server.use(http.post(AUTHORIZE_URL, () => HttpResponse.json(response)));

        await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toMatchObject({
          name: "DeviceFlowError",
          message:
            "Invalid response from device authorization endpoint: missing required fields",
        });
      },
    );

    it("wraps a failed authorization request in a DeviceFlowError", async () => {
      server.use(http.post(AUTHORIZE_URL, () => HttpResponse.error()));

      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        DeviceFlowError,
      );
      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: expect.stringContaining("Failed to start device flow:"),
      });
    });
  });

  describe("pollForToken", () => {
    it("returns token response on immediate success and forwards client headers", async () => {
      const mockTokenResponse = {
        access_token: "api-key-123",
        token_type: "Bearer",
      };

      let requestUrl: string | undefined;
      let requestBody: string | undefined;
      let requestHeaders: Headers | undefined;
      server.use(
        http.post(TOKEN_URL, async ({ request }) => {
          requestUrl = request.url;
          requestBody = await request.text();
          requestHeaders = request.headers;
          return HttpResponse.json(mockTokenResponse);
        }),
      );

      const result = await pollForToken(`${TEST_HOST_URL}///`, "device123", {
        interval: 5,
      });

      expect(result).toEqual({
        access_token: "api-key-123",
        token_type: "Bearer",
        expires_in: undefined,
      });
      expect(requestUrl).toBe(TOKEN_URL);
      expect(requestBody).toBe(
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code&device_code=device123",
      );
      expect(requestHeaders?.get("content-type")).toBe(
        "application/x-www-form-urlencoded",
      );
      // The wrapper must attach the agent-canvas observability headers.
      expectClientHeaders(requestHeaders as Headers);
    });

    it("defaults the token type when the successful response omits it", async () => {
      server.use(
        http.post(TOKEN_URL, () =>
          HttpResponse.json({ access_token: "api-key-123" }),
        ),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 5 }),
      ).resolves.toEqual({
        access_token: "api-key-123",
        token_type: "Bearer",
        expires_in: undefined,
      });
    });

    it("rejects a successful token response without an access token", async () => {
      server.use(
        http.post(TOKEN_URL, () => HttpResponse.json({ token_type: "Bearer" })),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 5 }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Invalid token response: missing access_token",
      });
    });

    it("waits for the configured interval before polling again", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          if (calls === 1) {
            return HttpResponse.json(
              {
                error: "authorization_pending",
                error_description: "User hasn't authorized yet",
              },
              { status: 400 },
            );
          }
          return HttpResponse.json({
            access_token: "api-key-123",
            token_type: "Bearer",
          });
        }),
      );

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      await vi.advanceTimersByTimeAsync(4999);
      expect(calls).toBe(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toBe(2);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("increases interval on slow_down error", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          if (calls === 1) {
            return HttpResponse.json(
              { error: "slow_down", interval: 7 },
              { status: 400 },
            );
          }
          return HttpResponse.json({
            access_token: "api-key-123",
            token_type: "Bearer",
          });
        }),
      );

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      await vi.advanceTimersByTimeAsync(6999);
      expect(calls).toBe(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toBe(2);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("throws on expired_token error", async () => {
      server.use(
        http.post(TOKEN_URL, () =>
          HttpResponse.json({ error: "expired_token" }, { status: 400 }),
        ),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Device code has expired. Please try again.",
        code: "expired_token",
      });
    });

    it("throws on access_denied error", async () => {
      server.use(
        http.post(TOKEN_URL, () =>
          HttpResponse.json({ error: "access_denied" }, { status: 400 }),
        ),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Authorization request was denied.",
        code: "access_denied",
      });
    });

    it("rejects a non-JSON token error response with its HTTP status", async () => {
      server.use(
        http.post(
          TOKEN_URL,
          () => new HttpResponse("<html>bad gateway</html>", { status: 502 }),
        ),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Unexpected response from server: 502",
      });
    });

    it.each([
      {
        description: "with its server description",
        error: "invalid_scope",
        errorDescription: "Requested scope is unavailable",
        expectedMessage:
          "Authorization error: invalid_scope - Requested scope is unavailable",
      },
      {
        description: "without a server description",
        error: "server_error",
        errorDescription: undefined,
        expectedMessage: "Authorization error: server_error",
      },
    ])(
      "preserves an unknown token error $description",
      async ({ error, errorDescription, expectedMessage }) => {
        server.use(
          http.post(TOKEN_URL, () =>
            HttpResponse.json(
              { error, error_description: errorDescription },
              { status: 400 },
            ),
          ),
        );

        await expect(
          pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
        ).rejects.toMatchObject({
          name: "DeviceFlowError",
          message: expectedMessage,
          code: error,
        });
      },
    );

    it("respects abort signal", async () => {
      const controller = new AbortController();

      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          return HttpResponse.json(
            { error: "authorization_pending" },
            { status: 400 },
          );
        }),
      );

      controller.abort();

      await expect(
        pollForToken(TEST_HOST_URL, "device123", {
          interval: 1,
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Authorization cancelled",
        code: "cancelled",
      });
      expect(calls).toBe(0);
    });

    it("cancels while waiting for the next poll", async () => {
      const controller = new AbortController();
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          return HttpResponse.json(
            { error: "authorization_pending" },
            { status: 400 },
          );
        }),
      );

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
        signal: controller.signal,
      });
      await vi.advanceTimersByTimeAsync(0);
      controller.abort();

      await expect(pollPromise).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Authorization cancelled",
        code: "cancelled",
      });
      expect(calls).toBe(1);
    });

    it("cancels when the signal aborts while a pending response is handled", async () => {
      const controller = new AbortController();
      // Aborting mid-request makes the SDK forward the cancellation to the
      // in-flight fetch, which logs a retry warning before the wait rejects.
      vi.spyOn(console, "warn").mockImplementation(() => {});
      server.use(
        http.post(TOKEN_URL, () => {
          controller.abort();
          return HttpResponse.json(
            { error: "authorization_pending" },
            { status: 400 },
          );
        }),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", {
          interval: 5,
          signal: controller.signal,
        }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Authorization cancelled",
        code: "cancelled",
      });
    });

    it("does not request a token when the timeout is already exhausted", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          return HttpResponse.json({ access_token: "api-key-123" });
        }),
      );

      await expect(
        pollForToken(TEST_HOST_URL, "device123", {
          interval: 1,
          timeout: 0,
        }),
      ).rejects.toMatchObject({
        name: "DeviceFlowError",
        message: "Timeout waiting for authorization. Please try again.",
        code: "timeout",
      });
      expect(calls).toBe(0);
    });

    it("caps slow_down interval at 30 seconds (DoS protection)", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          if (calls === 1) {
            return HttpResponse.json(
              { error: "slow_down", interval: 999999 },
              { status: 400 },
            );
          }
          return HttpResponse.json({
            access_token: "api-key-123",
            token_type: "Bearer",
          });
        }),
      );

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      await vi.advanceTimersByTimeAsync(29999);
      expect(calls).toBe(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toBe(2);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it.each([
      { description: "a numeric string", interval: "7" },
      { description: "zero", interval: 0 },
      { description: "an infinite number", interval: Number.POSITIVE_INFINITY },
    ])(
      "uses the RFC fallback for $description slow_down interval",
      async ({ interval }) => {
        let calls = 0;
        server.use(
          http.post(TOKEN_URL, () => {
            calls += 1;
            if (calls === 1) {
              return HttpResponse.json(
                { error: "slow_down", interval },
                { status: 400 },
              );
            }
            return HttpResponse.json({
              access_token: "api-key-123",
              token_type: "Bearer",
            });
          }),
        );

        const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
          interval: 5,
        });

        await vi.advanceTimersByTimeAsync(9999);
        expect(calls).toBe(1);

        await vi.advanceTimersByTimeAsync(1);
        expect(calls).toBe(2);

        await expect(pollPromise).resolves.toMatchObject({
          access_token: "api-key-123",
        });
      },
    );

    it("increments interval by 5 seconds per RFC 8628 when slow_down has no interval", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          if (calls === 1) {
            // No interval field - RFC 8628 mandates +5s increment
            return HttpResponse.json({ error: "slow_down" }, { status: 400 });
          }
          return HttpResponse.json({
            access_token: "api-key-123",
            token_type: "Bearer",
          });
        }),
      );

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5, // 5 seconds initial
      });

      await vi.advanceTimersByTimeAsync(9999);
      expect(calls).toBe(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toBe(2);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("continues polling on network errors instead of failing immediately", async () => {
      let calls = 0;
      server.use(
        http.post(TOKEN_URL, () => {
          calls += 1;
          // First call fails with a network error, second succeeds.
          if (calls === 1) {
            return HttpResponse.error();
          }
          return HttpResponse.json({
            access_token: "api-key-123",
            token_type: "Bearer",
          });
        }),
      );

      const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 1,
      });

      // Advance past the retry interval
      await vi.advanceTimersByTimeAsync(1000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
      expect(calls).toBe(2);
      expect(consoleSpy).toHaveBeenCalledWith(
        "Network error during polling, retrying:",
        expect.any(Error),
      );

      consoleSpy.mockRestore();
    });
  });
});
