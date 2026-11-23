import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getSandboxSelectionOptions,
  SANDBOX_SELECTION_PATH,
} from "#/api/cloud/sandbox-template-service.api";
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

describe("getSandboxSelectionOptions", () => {
  it("returns a disabled, empty payload when items is missing from an otherwise enabled response", async () => {
    fetchMock.mockResolvedValue(mockJsonResponse({ enabled: true }));

    const result = await getSandboxSelectionOptions(cloudBackend.id, "org-a");

    const [url] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}${SANDBOX_SELECTION_PATH}`);
    expect(result).toEqual({ enabled: false, items: [] });
  });

  it("returns a disabled, empty payload when items is not an array", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse({ enabled: true, items: "not-an-array" }),
    );

    const result = await getSandboxSelectionOptions(cloudBackend.id, "org-a");

    expect(result).toEqual({ enabled: false, items: [] });
  });

  it("passes through a valid payload, including an empty items array", async () => {
    fetchMock.mockResolvedValue(mockJsonResponse({ enabled: true, items: [] }));

    const result = await getSandboxSelectionOptions(cloudBackend.id, "org-a");

    expect(result).toEqual({ enabled: true, items: [] });
  });

  it("passes through a valid payload with items unchanged", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        enabled: true,
        items: [{ id: "large", display_name: "Monorepo Large" }],
      }),
    );

    const result = await getSandboxSelectionOptions(cloudBackend.id, "org-a");

    expect(result).toEqual({
      enabled: true,
      items: [{ id: "large", display_name: "Monorepo Large" }],
    });
  });

  it("throws when the backend is not a registered cloud backend", async () => {
    await expect(
      getSandboxSelectionOptions("missing-backend", "org-a"),
    ).rejects.toThrow("Sandbox templates require a registered cloud backend.");
  });
});
