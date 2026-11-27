import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  startDeviceFlow,
  pollForToken,
  isOpenHandsCloudHost,
  DeviceFlowError,
} from "../../src/api/device-flow-client";
import { AGENT_CANVAS_CLIENT_HEADERS } from "../../src/api/client-source";

const TEST_HOST_URL = "https://app.all-hands.dev";

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
    });

    it("returns false for invalid URLs", () => {
      expect(isOpenHandsCloudHost("")).toBe(false);
      expect(isOpenHandsCloudHost("not-a-url")).toBe(false);
    });
  });

  describe("startDeviceFlow", () => {
    it("returns device authorization response on success", async () => {
      const mockResponse = {
        device_code: "device123",
        user_code: "USER-1234",
        verification_uri: `${TEST_HOST_URL}/device`,
        verification_uri_complete: `${TEST_HOST_URL}/device?user_code=USER-1234`,
        expires_in: 600,
        interval: 5,
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await startDeviceFlow(TEST_HOST_URL);

      expect(result).toEqual(mockResponse);
      // Should call the cloud endpoint directly.
      const fetchCall = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(fetchCall[0]).toBe(`${TEST_HOST_URL}/oauth/device/authorize`);
      expect(fetchCall[1]).toEqual(expect.objectContaining({ method: "POST" }));

      const headers = new Headers(fetchCall[1].headers);
      expect(headers.get("Content-Type")).toBe("application/json");
      for (const [name, value] of Object.entries(AGENT_CANVAS_CLIENT_HEADERS)) {
        expect(headers.get(name)).toBe(value);
      }
    });

    it("normalizes host URL by removing trailing slashes", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            device_code: "dc",
            user_code: "uc",
            verification_uri: "v",
            verification_uri_complete: "vc",
            expires_in: 600,
            interval: 5,
          }),
      });

      await startDeviceFlow(`${TEST_HOST_URL}///`);

      // Verify the direct request targets the normalized host.
      const fetchCall = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(fetchCall[0]).toBe(`${TEST_HOST_URL}/oauth/device/authorize`);
    });

    it("throws DeviceFlowError on HTTP error", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve("Internal Server Error"),
      });

      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        DeviceFlowError,
      );
      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        /Failed to start device flow.*500/,
      );
    });

    it("throws DeviceFlowError on missing required fields", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            device_code: "dc",
            // Missing other required fields
          }),
      });

      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        DeviceFlowError,
      );
      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        /missing required fields/,
      );
    });

    it("throws DeviceFlowError on network error", async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error("Network failed"));

      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        DeviceFlowError,
      );
      await expect(startDeviceFlow(TEST_HOST_URL)).rejects.toThrow(
        /Network failed/,
      );
    });
  });

  describe("pollForToken", () => {
    it("returns token response on immediate success", async () => {
      const mockTokenResponse = {
        access_token: "api-key-123",
        token_type: "Bearer",
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockTokenResponse),
      });

      const result = await pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      expect(result).toEqual(mockTokenResponse);
      // Should call the cloud endpoint directly.
      const fetchCall = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(fetchCall[0]).toBe(`${TEST_HOST_URL}/oauth/device/token`);
      expect(fetchCall[1]).toEqual(expect.objectContaining({ method: "POST" }));

      const headers = new Headers(fetchCall[1].headers);
      expect(headers.get("Content-Type")).toBe(
        "application/x-www-form-urlencoded",
      );
      for (const [name, value] of Object.entries(AGENT_CANVAS_CLIENT_HEADERS)) {
        expect(headers.get(name)).toBe(value);
      }
    });

    it("polls until authorization is complete", async () => {
      const pendingResponse = {
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "authorization_pending",
            error_description: "User hasn't authorized yet",
          }),
      };
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      global.fetch = vi
        .fn()
        .mockResolvedValueOnce(pendingResponse)
        .mockResolvedValueOnce(successResponse);

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 1,
      });

      // Advance past the first poll interval
      await vi.advanceTimersByTimeAsync(1000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    it("increases interval on slow_down error", async () => {
      const slowDownResponse = {
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "slow_down",
            interval: 10,
          }),
      };
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      global.fetch = vi
        .fn()
        .mockResolvedValueOnce(slowDownResponse)
        .mockResolvedValueOnce(successResponse);

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      // Advance by new interval (10 seconds)
      await vi.advanceTimersByTimeAsync(10000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("throws on expired_token error", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "expired_token",
          }),
      });

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toThrow(DeviceFlowError);
      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toThrow(/expired/i);
    });

    it("throws on access_denied error", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "access_denied",
          }),
      });

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toThrow(DeviceFlowError);
      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toThrow(/denied/i);
    });

    it("reports a non-JSON error response instead of retrying it as a network error", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: () => Promise.reject(new SyntaxError("invalid JSON")),
      });

      await expect(
        pollForToken(TEST_HOST_URL, "device123", { interval: 1 }),
      ).rejects.toThrow(/Unexpected response from server: 502/);
    });

    it("respects abort signal", async () => {
      vi.useRealTimers(); // Use real timers for this test
      const controller = new AbortController();

      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "authorization_pending",
          }),
      });

      // Pre-abort the controller
      controller.abort();

      // Now the promise should reject immediately with cancelled
      await expect(
        pollForToken(TEST_HOST_URL, "device123", {
          interval: 1,
          signal: controller.signal,
        }),
      ).rejects.toThrow(/cancelled/i);
    });

    it("reports cancellation when aborted between polling attempts", async () => {
      const controller = new AbortController();
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: "authorization_pending" }),
      });

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
        signal: controller.signal,
      });
      const rejection = expect(pollPromise).rejects.toMatchObject({
        code: "cancelled",
      });

      await vi.advanceTimersByTimeAsync(0);
      controller.abort();

      await rejection;
    });

    it("times out after specified duration", async () => {
      vi.useRealTimers(); // Use real timers for this test
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "authorization_pending",
          }),
      });

      // Use very short timeout
      await expect(
        pollForToken(TEST_HOST_URL, "device123", {
          interval: 0.01, // 10ms interval
          timeout: 50, // 50ms timeout
        }),
      ).rejects.toThrow(/timeout/i);
    }, 10000);

    it("caps slow_down interval at 30 seconds (DoS protection)", async () => {
      const slowDownResponse = {
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "slow_down",
            interval: 999999, // Malicious server tries to DoS
          }),
      };
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      global.fetch = vi
        .fn()
        .mockResolvedValueOnce(slowDownResponse)
        .mockResolvedValueOnce(successResponse);

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      // Should use 30s max, not 999999s
      await vi.advanceTimersByTimeAsync(30000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    it("rejects non-numeric slow_down interval (type confusion protection)", async () => {
      const slowDownResponse = {
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "slow_down",
            interval: "pwned", // Non-numeric value
          }),
      };
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      global.fetch = vi
        .fn()
        .mockResolvedValueOnce(slowDownResponse)
        .mockResolvedValueOnce(successResponse);

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5,
      });

      // With invalid interval, should use RFC 8628 default: current + 5s
      // Starting interval is 5s, so next should be 10s (5000 + 5000 = 10000ms)
      await vi.advanceTimersByTimeAsync(10000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("increments interval by 5 seconds per RFC 8628 when slow_down has no interval", async () => {
      const slowDownResponse = {
        ok: false,
        status: 400,
        json: () =>
          Promise.resolve({
            error: "slow_down",
            // No interval field - RFC 8628 mandates +5s increment
          }),
      };
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      global.fetch = vi
        .fn()
        .mockResolvedValueOnce(slowDownResponse)
        .mockResolvedValueOnce(successResponse);

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 5, // 5 seconds initial
      });

      // RFC 8628: must increment by 5 seconds, so 5s -> 10s
      await vi.advanceTimersByTimeAsync(10000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
    });

    it("continues polling on network errors instead of failing immediately", async () => {
      const networkError = new Error("Network failed");
      const successResponse = {
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            access_token: "api-key-123",
            token_type: "Bearer",
          }),
      };

      // First call fails with network error, second succeeds
      global.fetch = vi
        .fn()
        .mockRejectedValueOnce(networkError)
        .mockResolvedValueOnce(successResponse);

      const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const pollPromise = pollForToken(TEST_HOST_URL, "device123", {
        interval: 1,
      });

      // Advance past the retry interval
      await vi.advanceTimersByTimeAsync(1000);

      const result = await pollPromise;
      expect(result.access_token).toBe("api-key-123");
      expect(consoleSpy).toHaveBeenCalledWith(
        "Network error during polling, retrying:",
        networkError,
      );

      consoleSpy.mockRestore();
    });
  });
});
