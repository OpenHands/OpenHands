import { vi } from "vitest";

type FetchMock = ReturnType<typeof vi.fn>;

export function mockJsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    statusText: status === 200 ? "OK" : "Error",
    headers: { "content-type": "application/json" },
  });
}

export function mockBlobResponse(
  body: BodyInit,
  contentType: string,
): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": contentType },
  });
}

export function getFetchCall(
  fetchMock: FetchMock,
  index = 0,
): [string, RequestInit] {
  const [url, init] = fetchMock.mock.calls[index] as [string, RequestInit];
  return [url, init ?? {}];
}

export function getJsonBody(init: RequestInit): Record<string, unknown> {
  return JSON.parse(String(init.body ?? "{}")) as Record<string, unknown>;
}
