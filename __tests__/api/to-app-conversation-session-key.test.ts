import { afterEach, describe, expect, it, vi } from "vitest";

import { toAppConversation } from "#/api/agent-server-adapter";
import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";

const directInfo = (id: string) => ({
  id,
  created_at: "2026-05-05T00:00:00Z",
  updated_at: "2026-05-05T00:00:00Z",
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.unstubAllEnvs();
});

describe("toAppConversation session_api_key hydration", () => {
  it("uses the active backend registry apiKey instead of VITE_SESSION_API_KEY", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");

    setRegisteredBackends([
      {
        id: SEEDED_DEFAULT_BACKEND_ID,
        name: "Local",
        host: window.location.origin,
        apiKey: "stale-session-key",
        kind: "local",
      },
    ]);

    const conversation = toAppConversation(directInfo("conv-1"));
    expect(conversation.session_api_key).toBe("stale-session-key");
  });
});
