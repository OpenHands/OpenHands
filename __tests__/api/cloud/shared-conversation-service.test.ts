import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudSharedConversation,
  searchCloudSharedEvents,
} from "#/api/cloud/shared-conversation-service.api";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

function activateCloudBackend() {
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("getCloudSharedConversation", () => {
  it("GETs /api/shared-conversations for the id and returns the matching conversation", async () => {
    // Arrange
    activateCloudBackend();
    fetchMock.mockResolvedValue(
      mockJsonResponse([{ id: "conv-abc", created_by_user_id: "user-1" }]),
    );

    // Act
    const conversation = await getCloudSharedConversation("conv-abc");

    // Assert
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/shared-conversations?ids=conv-abc`,
    );
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(conversation).toEqual({
      id: "conv-abc",
      created_by_user_id: "user-1",
    });
  });

  it("returns null when the backend does not share the conversation with the caller", async () => {
    // Arrange
    activateCloudBackend();
    fetchMock.mockResolvedValue(mockJsonResponse([null]));

    // Act
    const conversation = await getCloudSharedConversation("conv-abc");

    // Assert
    expect(conversation).toBeNull();
  });

  it("rejects without calling the cloud API when the active backend is local", async () => {
    // Arrange — the default state after reset is the bundled local backend.
    // Act / Assert
    await expect(getCloudSharedConversation("conv-abc")).rejects.toThrow(
      /cloud backend/,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("searchCloudSharedEvents", () => {
  it("GETs /api/shared-events/search with the conversation id, page size and cursor", async () => {
    // Arrange
    activateCloudBackend();
    fetchMock.mockResolvedValue(
      mockJsonResponse({ items: [], next_page_id: null }),
    );

    // Act
    const page = await searchCloudSharedEvents({
      conversationId: "conv-abc",
      limit: 50,
      pageId: "cursor-2",
    });

    // Assert
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/shared-events/search?conversation_id=conv-abc&limit=50&page_id=cursor-2`,
    );
    expect(init).toMatchObject({ method: "GET" });
    expect(page).toEqual({ items: [], next_page_id: null });
  });
});
