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

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@openhands/typescript-client/clients")
    >();

  return {
    ...actual,
    ConversationClient: vi.fn().mockImplementation(() => ({
      updateConversation: vi
        .fn()
        .mockRejectedValue(
          new Error("Cloud title updates must not use ConversationClient"),
        ),
    })),
  };
});

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
    mockJsonResponse({ id: "conv-abc", title: "Renamed conversation" }),
  );
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("AgentServerConversationService.updateConversationTitle", () => {
  it("PATCHes the app-conversation directly on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const result = await AgentServerConversationService.updateConversationTitle(
      "conv-abc",
      "Renamed conversation",
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/app-conversations/conv-abc`);
    expect(init).toMatchObject({
      method: "PATCH",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual({ title: "Renamed conversation" });
    expect(result).toMatchObject({
      id: "conv-abc",
      title: "Renamed conversation",
    });
  });
});
