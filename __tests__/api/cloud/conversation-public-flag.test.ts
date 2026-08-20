import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  getFetchCall,
  getJsonBody,
  mockJsonResponse,
} from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(
    mockJsonResponse({ id: "conv-abc", public: true }),
  );
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("AgentServerConversationService.updateConversationPublicFlag", () => {
  it("PATCHes /api/v1/app-conversations/{id} directly on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await AgentServerConversationService.updateConversationPublicFlag(
      "conv-abc",
      true,
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/app-conversations/conv-abc`);
    expect(init).toMatchObject({
      method: "PATCH",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual({ public: true });
  });

  it("rejects without calling the cloud API when the active backend is local", async () => {
    // Default state after reset is the bundled local backend.
    await expect(
      AgentServerConversationService.updateConversationPublicFlag(
        "conv-abc",
        true,
      ),
    ).rejects.toThrow(/cloud backend/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
