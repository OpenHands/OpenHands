import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  buildHttpBaseUrlMock,
  callCloudProxyMock,
  conversationClientConstructorMock,
  getActiveBackendMock,
  getAgentServerClientOptionsMock,
  getAgentServerHttpClientOptionsMock,
  getEventCountMock,
  remoteEventsListConstructorMock,
  remoteSearchMock,
  respondToConfirmationMock,
} = vi.hoisted(() => ({
  buildHttpBaseUrlMock: vi.fn(),
  callCloudProxyMock: vi.fn(),
  conversationClientConstructorMock: vi.fn(),
  getActiveBackendMock: vi.fn(),
  getAgentServerClientOptionsMock: vi.fn(),
  getAgentServerHttpClientOptionsMock: vi.fn(),
  getEventCountMock: vi.fn(),
  remoteEventsListConstructorMock: vi.fn(),
  remoteSearchMock: vi.fn(),
  respondToConfirmationMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: class {
    constructor(options: unknown) {
      conversationClientConstructorMock(options);
    }

    respondToConfirmation = respondToConfirmationMock;
    getEventCount = getEventCountMock;
  },
}));
vi.mock("@openhands/typescript-client/events/remote-events-list", () => ({
  RemoteEventsList: class {
    constructor(options: unknown, conversationId: string) {
      remoteEventsListConstructorMock(options, conversationId);
    }

    search = remoteSearchMock;
  },
}));
vi.mock("#/utils/websocket-url", () => ({
  buildHttpBaseUrl: buildHttpBaseUrlMock,
}));
vi.mock("../backend-registry/active-store", () => ({
  getActiveBackend: getActiveBackendMock,
}));
vi.mock("../cloud/proxy", () => ({ callCloudProxy: callCloudProxyMock }));
vi.mock("../agent-server-client-options", () => ({
  getAgentServerClientOptions: getAgentServerClientOptionsMock,
  getAgentServerHttpClientOptions: getAgentServerHttpClientOptionsMock,
}));

import EventService from "./event-service.api";

const cloudBackend = {
  id: "cloud-backend",
  name: "Cloud",
  host: "https://app.example.com",
  apiKey: "cloud-api-key",
  kind: "cloud",
};

const localBackend = {
  id: "local-backend",
  name: "Local",
  host: "http://localhost:3000",
  apiKey: "local-api-key",
  kind: "local",
};

function useBackend(backend: typeof cloudBackend | typeof localBackend): void {
  getActiveBackendMock.mockReturnValue({ backend, orgId: null });
}

describe("EventService", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    useBackend(cloudBackend);
    buildHttpBaseUrlMock.mockReturnValue("https://runtime.example.com/base");
    getAgentServerClientOptionsMock.mockReturnValue({
      host: "http://local-client.example.com",
      apiKey: "client-key",
      workingDir: "workspace/project",
    });
    getAgentServerHttpClientOptionsMock.mockReturnValue({
      baseUrl: "http://local-http.example.com",
      apiKey: "http-key",
      timeout: 60_000,
    });
  });

  describe("respondToConfirmation", () => {
    it("routes cloud confirmations through the runtime proxy with session authentication", async () => {
      const request = { accept: false, reason: "Needs human review" };
      callCloudProxyMock.mockResolvedValue({ success: true });

      await expect(
        EventService.respondToConfirmation(
          "conversation-cloud",
          "wss://runtime.example.com/base/api/conversations/conversation-cloud",
          request,
          "session-key",
        ),
      ).resolves.toEqual({ success: true });

      expect(buildHttpBaseUrlMock).toHaveBeenCalledWith(
        "wss://runtime.example.com/base/api/conversations/conversation-cloud",
      );
      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "POST",
        hostOverride: "https://runtime.example.com/base",
        path: "/api/conversations/conversation-cloud/events/respond_to_confirmation",
        body: request,
        authMode: "session-api-key",
        sessionApiKey: "session-key",
      });
      expect(conversationClientConstructorMock).not.toHaveBeenCalled();
    });

    it("uses the typed conversation client for local confirmations", async () => {
      useBackend(localBackend);
      respondToConfirmationMock.mockResolvedValue({ success: false });

      await expect(
        EventService.respondToConfirmation(
          "conversation-local",
          "http://localhost:3000/api/conversations/conversation-local",
          { accept: true },
        ),
      ).resolves.toEqual({ success: false });

      expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith({
        conversationUrl:
          "http://localhost:3000/api/conversations/conversation-local",
        sessionApiKey: undefined,
      });
      expect(conversationClientConstructorMock).toHaveBeenCalledWith({
        host: "http://local-client.example.com",
        apiKey: "client-key",
        workingDir: "workspace/project",
      });
      expect(respondToConfirmationMock).toHaveBeenCalledWith(
        "conversation-local",
        { accept: true },
      );
      expect(callCloudProxyMock).not.toHaveBeenCalled();
    });
  });

  describe("getEventCount", () => {
    it("gets the cloud count from the runtime proxy", async () => {
      callCloudProxyMock.mockResolvedValue(17);

      await expect(
        EventService.getEventCount(
          "conversation-cloud",
          "https://runtime.example.com/api/conversations/conversation-cloud",
          null,
        ),
      ).resolves.toBe(17);

      expect(buildHttpBaseUrlMock).toHaveBeenCalledWith(
        "https://runtime.example.com/api/conversations/conversation-cloud",
      );
      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        hostOverride: "https://runtime.example.com/base",
        path: "/api/conversations/conversation-cloud/events/count",
        authMode: "session-api-key",
        sessionApiKey: null,
      });
      expect(conversationClientConstructorMock).not.toHaveBeenCalled();
    });

    it("gets the local count from the typed conversation client", async () => {
      useBackend(localBackend);
      getEventCountMock.mockResolvedValue(8);

      await expect(
        EventService.getEventCount(
          "conversation-local",
          "http://localhost:3000/api/conversations/conversation-local",
          "session-key",
        ),
      ).resolves.toBe(8);

      expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith({
        conversationUrl:
          "http://localhost:3000/api/conversations/conversation-local",
        sessionApiKey: "session-key",
      });
      expect(conversationClientConstructorMock).toHaveBeenCalledWith({
        host: "http://local-client.example.com",
        apiKey: "client-key",
        workingDir: "workspace/project",
      });
      expect(getEventCountMock).toHaveBeenCalledWith("conversation-local");
      expect(callCloudProxyMock).not.toHaveBeenCalled();
    });
  });

  describe("searchEvents", () => {
    it("caps cloud limits and forwards every supported history filter", async () => {
      const event = { id: "event-1" };
      callCloudProxyMock.mockResolvedValue({
        items: [event],
        next_page_id: "page-2",
      });

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          limit: 250,
          pageId: "page-1",
          sortOrder: "TIMESTAMP_DESC",
          timestampGte: "2026-07-12T00:00:00.000Z",
          timestampLt: "2026-07-13T00:00:00.000Z",
        }),
      ).resolves.toEqual({ items: [event], next_page_id: "page-2" });

      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/v1/conversation/conversation-cloud/events/search?limit=100&sort_order=TIMESTAMP_DESC&page_id=page-1&timestamp__gte=2026-07-12T00%3A00%3A00.000Z&timestamp__lt=2026-07-13T00%3A00%3A00.000Z",
      });
    });

    it("uses default cloud options and normalizes a missing response", async () => {
      callCloudProxyMock.mockResolvedValue(undefined);

      await expect(
        EventService.searchEvents("conversation-cloud"),
      ).resolves.toEqual({ items: [], next_page_id: null });

      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/v1/conversation/conversation-cloud/events/search?limit=100",
      });
    });

    it("preserves zero as an explicit cloud limit", async () => {
      callCloudProxyMock.mockResolvedValue({});

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          limit: 0,
        }),
      ).resolves.toEqual({ items: [], next_page_id: null });

      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/v1/conversation/conversation-cloud/events/search?limit=0",
      });
    });

    it("rethrows cloud failures when no pagination filter was requested", async () => {
      const searchError = new Error("cloud unavailable");
      callCloudProxyMock.mockRejectedValue(searchError);

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          limit: 25,
        }),
      ).rejects.toBe(searchError);
    });

    it("rethrows filtered cloud failures for completeness-sensitive callers", async () => {
      const paginationError = new Error("pagination unsupported");
      callCloudProxyMock.mockRejectedValue(paginationError);

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          pageId: "page-2",
          strictPagination: true,
        }),
      ).rejects.toBe(paginationError);
    });

    it("retains the empty-page fallback for ordinary cloud pagination", async () => {
      callCloudProxyMock.mockRejectedValue(new Error("pagination unsupported"));
      const warn = vi
        .spyOn(console, "warn")
        .mockImplementation(() => undefined);

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          timestampLt: "2026-07-10T12:34:56.000Z",
        }),
      ).resolves.toEqual({ items: [], next_page_id: null });

      expect(warn).toHaveBeenCalledWith(
        "[EventService] Cloud backend doesn't support pagination filters. " +
          "Falling back to initial load only. " +
          "Server needs OpenHands/OpenHands#14399.",
      );
      warn.mockRestore();
    });

    it("treats a sort-only cloud failure as unsupported filtered pagination", async () => {
      callCloudProxyMock.mockRejectedValue(new Error("pagination unsupported"));
      const warn = vi
        .spyOn(console, "warn")
        .mockImplementation(() => undefined);

      await expect(
        EventService.searchEvents("conversation-cloud", null, null, {
          sortOrder: "TIMESTAMP_DESC",
        }),
      ).resolves.toEqual({ items: [], next_page_id: null });

      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/v1/conversation/conversation-cloud/events/search?limit=100&sort_order=TIMESTAMP_DESC",
      });
      expect(warn).toHaveBeenCalledOnce();
      warn.mockRestore();
    });

    it("recognizes a lower timestamp bound as a cloud pagination filter", async () => {
      callCloudProxyMock.mockResolvedValue({
        items: [],
        next_page_id: null,
      });

      await EventService.searchEvents("conversation-cloud", null, null, {
        timestampGte: "2026-07-10T12:34:56.000Z",
      });

      expect(callCloudProxyMock).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/v1/conversation/conversation-cloud/events/search?limit=100&timestamp__gte=2026-07-10T12%3A34%3A56.000Z",
      });
    });

    it("passes all local filters to the remote events list", async () => {
      useBackend(localBackend);
      const event = { id: "event-local" };
      remoteSearchMock.mockResolvedValue({
        items: [event],
        next_page_id: "local-page-2",
      });

      await expect(
        EventService.searchEvents(
          "conversation-local",
          "http://localhost:3000/api/conversations/conversation-local",
          "session-key",
          {
            limit: 40,
            pageId: "local-page-1",
            sortOrder: "TIMESTAMP",
            timestampGte: "2026-07-11T00:00:00.000Z",
            timestampLt: "2026-07-12T00:00:00.000Z",
          },
        ),
      ).resolves.toEqual({
        items: [event],
        next_page_id: "local-page-2",
      });

      expect(getAgentServerHttpClientOptionsMock).toHaveBeenCalledWith({
        conversationUrl:
          "http://localhost:3000/api/conversations/conversation-local",
        sessionApiKey: "session-key",
      });
      expect(remoteEventsListConstructorMock).toHaveBeenCalledWith(
        {
          baseUrl: "http://local-http.example.com",
          apiKey: "http-key",
          timeout: 60_000,
        },
        "conversation-local",
      );
      expect(remoteSearchMock).toHaveBeenCalledWith({
        limit: 40,
        page_id: "local-page-1",
        sort_order: "TIMESTAMP",
        timestamp__gte: "2026-07-11T00:00:00.000Z",
        timestamp__lt: "2026-07-12T00:00:00.000Z",
      });
      expect(callCloudProxyMock).not.toHaveBeenCalled();
    });

    it("uses default local options and normalizes a missing page", async () => {
      useBackend(localBackend);
      remoteSearchMock.mockResolvedValue(undefined);

      await expect(
        EventService.searchEvents("conversation-local"),
      ).resolves.toEqual({ items: [], next_page_id: null });

      expect(getAgentServerHttpClientOptionsMock).toHaveBeenCalledWith({
        conversationUrl: undefined,
        sessionApiKey: undefined,
      });
      expect(remoteSearchMock).toHaveBeenCalledWith({ limit: 100 });
    });
  });
});
