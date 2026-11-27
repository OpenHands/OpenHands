import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { batchGetCloudSandboxes } from "#/api/cloud/sandbox-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "cloud-prod",
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
  fetchMock.mockResolvedValue(mockJsonResponse([]));
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("batchGetCloudSandboxes", () => {
  it("targets /api/v1/sandboxes with one id query param per sandbox id", async () => {
    // Arrange — multiple ids exercises the URLSearchParams.append path,
    // which is the cloud contract for batch-fetching sandboxes (the GUI
    // reads sandbox.exposed_urls from the response to find the VSCODE
    // URL instead of asking the runtime for a localhost address).
    const ids = ["sandbox-a", "sandbox-b"];

    // Act
    await batchGetCloudSandboxes(ids);

    const [url, init] = getFetchCall(fetchMock);
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(url).toBe(
      `${cloudBackend.host}/api/v1/sandboxes?id=sandbox-a&id=sandbox-b`,
    );
  });
});
