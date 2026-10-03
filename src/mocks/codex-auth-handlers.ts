import { http, HttpResponse } from "msw";
import type { CodexAuthStatus } from "#/api/codex-auth-service";

const MOCK_AUTH_PATH = "*/api/acp/codex/auth";
let connected = false;
let pending = false;

export function resetMockCodexAuth() {
  connected = false;
  pending = false;
}

const status = (): CodexAuthStatus => ({
  connected,
  state: connected ? "connected" : "disconnected",
  expires_at: null,
});

export const CODEX_AUTH_HANDLERS = [
  http.get(`${MOCK_AUTH_PATH}/status`, () => HttpResponse.json(status())),
  http.post(`${MOCK_AUTH_PATH}/device/start`, () => {
    pending = true;
    return HttpResponse.json({
      device_code: "mock-codex-handle",
      user_code: "DEMO-CODE",
      verification_uri: "https://example.test/codex-device-login",
      expires_at: Date.now() + 900000,
      interval_seconds: 1,
    });
  }),
  http.post(`${MOCK_AUTH_PATH}/device/poll`, () => {
    if (pending) {
      pending = false;
      connected = true;
    }
    return HttpResponse.json(status());
  }),
  http.post(`${MOCK_AUTH_PATH}/device/cancel`, () => {
    pending = false;
    return HttpResponse.json({ ...status(), state: "cancelled" });
  }),
  http.post(`${MOCK_AUTH_PATH}/logout`, () => {
    resetMockCodexAuth();
    return HttpResponse.json(status());
  }),
];
