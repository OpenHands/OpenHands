import { beforeEach, describe, expect, it, vi } from "vitest";

const { callCloudProxyMock } = vi.hoisted(() => ({
  callCloudProxyMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: class {},
}));
vi.mock("@openhands/typescript-client/events/remote-events-list", () => ({
  RemoteEventsList: class {},
}));
vi.mock("../backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: "cloud" } }),
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
