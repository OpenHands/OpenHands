import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { fetchCloudConversationHooks } from "#/api/cloud/hooks-service.api";
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
  global.fetch = originalFetch;
});

describe("fetchCloudConversationHooks", () => {
  it("returns the existing GetHooksResponse from the first-class endpoint", async () => {
    const response = {
      hooks: [
        {
          event_type: "pre_tool_use",
          matchers: [
            {
              matcher: "*",
              hooks: [
                {
                  type: "command",
                  command: "lint",
                  timeout: 60,
                  async: false,
                },
              ],
            },
          ],
        },
      ],
    };
    fetchMock.mockResolvedValue(mockJsonResponse(response));

    await expect(
      fetchCloudConversationHooks("conversation/1", 30),
    ).resolves.toEqual(response);

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      "https://app.all-hands.dev/api/v1/app-conversations/conversation%2F1/hooks",
    );
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});
