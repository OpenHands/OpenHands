import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { testCloudMcpServer } from "#/api/cloud/mcp-service.api";
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
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("testCloudMcpServer", () => {
  it("posts the probe request to /api/v1/mcp/test on the active cloud backend", async () => {
    // Arrange
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        ok: false,
        error: "refused",
        error_kind: "connection",
      }),
    );
    const request = {
      name: "jira",
      server: {
        type: "http" as const,
        url: "https://mcp-jira.example.com/mcp",
      },
      timeout: 15,
    };

    // Act
    const result = await testCloudMcpServer(request);

    // Assert
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/mcp/test`);
    expect(init).toMatchObject({
      method: "POST",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual(request);
    expect(result).toEqual({
      ok: false,
      error: "refused",
      error_kind: "connection",
    });
  });
});
