import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudSharedConversation,
  searchCloudSharedConversations,
  searchCloudSharedEvents,
} from "#/api/cloud/shared-conversation-service.api";
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

describe("searchCloudSharedConversations", () => {
  it("GETs /api/shared-conversations/search with the page size and cursor and returns the page", async () => {
    // Arrange
    activateCloudBackend();
    const page = {
      items: [{ id: "conv-abc", created_by_user_id: "user-1" }],
      next_page_id: "40",
    };
    fetchMock.mockResolvedValue(mockJsonResponse(page));

    // Act
    const result = await searchCloudSharedConversations({
      limit: 20,
      pageId: "20",
    });

    // Assert
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/shared-conversations/search?limit=20&page_id=20`,
    );
    expect(init).toMatchObject({ method: "GET" });
    expect(result).toEqual(page);
  });

  it("returns an empty page when the backend does not serve the listing", async () => {
    // Arrange — a backend older than the listing answers the unknown API path
    // with its HTML page instead of JSON.
    activateCloudBackend();
    fetchMock.mockResolvedValue(
      mockBlobResponse("<!doctype html><html></html>", "text/html"),
    );

    // Act
    const result = await searchCloudSharedConversations({});

    // Assert
    expect(result).toEqual({ items: [], next_page_id: null });
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
