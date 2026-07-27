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
  mockBlobResponse,
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
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(mockJsonResponse({}));
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("AgentServerConversationService.downloadConversation cloud branch", () => {
  it("calls the cloud download endpoint directly with responseType blob and returns the Blob", async () => {
    fetchMock.mockResolvedValueOnce(
      mockBlobResponse("zip-bytes", "application/zip"),
    );

    const result =
      await AgentServerConversationService.downloadConversation("conv-abc");

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/v1/app-conversations/conv-abc/download`,
    );
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    await expect(result.text()).resolves.toBe("zip-bytes");
  });
});
