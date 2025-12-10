import axios from "axios";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { buildSessionHeaders } from "#/utils/utils";

import EventService from "#/api/event-service/event-service.api";

vi.mock("axios", () => {
  const get = vi.fn();
  return { default: { get } };
});

vi.mock("#/utils/websocket-url", () => ({
  buildHttpBaseUrl: vi.fn(),
}));

vi.mock("#/utils/utils", () => ({
  buildSessionHeaders: vi.fn(),
}));

const mockedAxios = axios as unknown as { get: ReturnType<typeof vi.fn> };

describe("EventService.getEventCount", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls the runtime event count endpoint with headers", async () => {
    const runtimeUrl = "http://runtime-url";
    const headers = { Authorization: "Bearer session" };
    const conversationId = "abc-123";
    const conversationUrl = "http://example.com/api/conversations/abc-123";
    const sessionApiKey = "session-key";

    vi.mocked(buildHttpBaseUrl).mockReturnValue(runtimeUrl);
    vi.mocked(buildSessionHeaders).mockReturnValue(headers);
    mockedAxios.get.mockResolvedValue({ data: 42 });

    const result = await EventService.getEventCount(
      conversationId,
      conversationUrl,
      sessionApiKey,
    );

    expect(buildHttpBaseUrl).toHaveBeenCalledWith(conversationUrl);
    expect(buildSessionHeaders).toHaveBeenCalledWith(sessionApiKey);
    expect(mockedAxios.get).toHaveBeenCalledWith(
      `${runtimeUrl}/api/conversations/${conversationId}/events/count`,
      { headers },
    );
    expect(result).toBe(42);
  });

  it('returns Unauthorized detail when Authorization header is "none"', async () => {
    const runtimeUrl = "http://runtime-url";
    const headers = { Authorization: "none" };
    const conversationId = "abc-123";
    const conversationUrl = "http://example.com/api/conversations/abc-123";

    vi.mocked(buildHttpBaseUrl).mockReturnValue(runtimeUrl);
    vi.mocked(buildSessionHeaders).mockReturnValue(headers);
    mockedAxios.get.mockResolvedValue({ data: { detail: "Unauthorized" } });

    const result = await EventService.getEventCount(
      conversationId,
      conversationUrl,
      undefined,
    );

    expect(buildHttpBaseUrl).toHaveBeenCalledWith(conversationUrl);
    expect(buildSessionHeaders).toHaveBeenCalledWith(undefined);
    expect(mockedAxios.get).toHaveBeenCalledWith(
      `${runtimeUrl}/api/conversations/${conversationId}/events/count`,
      { headers },
    );
    expect(result).toEqual({ detail: "Unauthorized" });
  });
});
