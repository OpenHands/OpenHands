import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { callCloudProxyMock, activeBackendMock, remoteSearchMock } = vi.hoisted(
  () => ({
    callCloudProxyMock: vi.fn(),
    activeBackendMock: vi.fn(() => ({ backend: { kind: "cloud" } })),
    remoteSearchMock: vi.fn(),
  }),
);

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: class {},
}));
vi.mock("@openhands/typescript-client/events/remote-events-list", () => ({
  RemoteEventsList: class {
    search = remoteSearchMock;
  },
}));
vi.mock("../backend-registry/active-store", () => ({
  getActiveBackend: () => activeBackendMock(),
}));
vi.mock("../cloud/proxy", () => ({ callCloudProxy: callCloudProxyMock }));
vi.mock("../agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn(),
  getAgentServerHttpClientOptions: vi.fn(),
}));

import EventService from "./event-service.api";

describe("EventService.searchEvents strict pagination", () => {
  beforeEach(() => {
    callCloudProxyMock.mockReset();
  });

  it("rethrows unsupported cloud pagination for completeness-sensitive callers", async () => {
    const paginationError = new Error("pagination unsupported");
    callCloudProxyMock.mockRejectedValue(paginationError);

    await expect(
      EventService.searchEvents("conversation-1", null, null, {
        limit: 100,
        sortOrder: "TIMESTAMP_DESC",
        strictPagination: true,
      }),
    ).rejects.toBe(paginationError);
  });

  it("retains the empty-page fallback for ordinary chat pagination", async () => {
    callCloudProxyMock.mockRejectedValue(new Error("pagination unsupported"));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(
      EventService.searchEvents("conversation-1", null, null, {
        limit: 50,
        timestampLt: "2026-07-10T12:34:56.000Z",
      }),
    ).resolves.toEqual({ items: [], next_page_id: null });

    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });
});

describe("EventService.searchEvents local path", () => {
  beforeEach(() => {
    activeBackendMock.mockReturnValue({ backend: { kind: "local" } });
    remoteSearchMock.mockReset();
  });

  afterEach(() => {
    activeBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
  });

  it("narrows the canonical typed-client page via isAgentServerEvent", async () => {
    const wellFormed = {
      id: "evt-1",
      timestamp: "2026-07-10T12:00:00.000Z",
      source: "user",
      kind: "MessageEvent",
      llm_message: {
        role: "user",
        content: [{ type: "text", text: "hello" }],
      },
    };
    const malformed = { nope: true };
    remoteSearchMock.mockResolvedValue({
      items: [wellFormed, malformed],
      next_page_id: "p2",
    });

    await expect(
      EventService.searchEvents(
        "conversation-1",
        "http://localhost:3000",
        null,
      ),
    ).resolves.toEqual({ items: [wellFormed], next_page_id: "p2" });

    // A valid wire envelope keeps its next-page cursor for pagination.
    expect(remoteSearchMock).toHaveBeenCalledWith({ limit: 100 });
  });
});
