import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
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

const localBackend: Backend = {
  id: "default-local",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "local-key",
  kind: "local",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  fetchMock.mockImplementation((url, init) => {
    if (init?.method === "PATCH") {
      return Promise.resolve(
        mockJsonResponse({
          id: "conv-repo",
          title: "Test conversation",
          selected_repository: "OpenHands/OpenHands",
          selected_branch: "main",
          git_provider: "github",
        }),
      );
    }
    // GET /api/v1/app-conversations?ids=... returns an array
    return Promise.resolve(
      mockJsonResponse([
        {
          id: "conv-repo",
          title: "Test conversation",
          selected_repository: "OpenHands/OpenHands",
          selected_branch: "main",
          git_provider: "github",
        },
      ]),
    );
  });
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("AgentServerConversationService.updateConversationRepository on Cloud", () => {
  it("PATCHes the app-conversation with selected_repository, selected_branch, and git_provider on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const result =
      await AgentServerConversationService.updateConversationRepository(
        "conv-repo",
        "OpenHands/OpenHands",
        "main",
        "github",
      );

    expect(fetchMock).toHaveBeenCalled();
    const patchCall = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "PATCH",
    );
    expect(patchCall).toBeDefined();
    const [url, init] = patchCall!;
    expect(url).toBe(`${cloudBackend.host}/api/v1/app-conversations/conv-repo`);
    expect(init).toMatchObject({
      method: "PATCH",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(getJsonBody(init)).toEqual({
      selected_repository: "OpenHands/OpenHands",
      selected_branch: "main",
      git_provider: "github",
    });
    expect(result).toMatchObject({
      id: "conv-repo",
      selected_repository: "OpenHands/OpenHands",
      selected_branch: "main",
      git_provider: "github",
    });
  });

  it("PATCHes the app-conversation with nulls when clearing repository on a cloud backend", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await AgentServerConversationService.updateConversationRepository(
      "conv-repo",
      null,
    );

    const patchCall = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "PATCH",
    );
    expect(patchCall).toBeDefined();
    const [url, init] = patchCall!;
    expect(url).toBe(`${cloudBackend.host}/api/v1/app-conversations/conv-repo`);
    expect(getJsonBody(init)).toEqual({
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    });
  });

  it("tolerates server rejection and falls back cleanly without breaking local flow", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    fetchMock.mockImplementation((url, init) => {
      if (init?.method === "PATCH") {
        return Promise.reject(new Error("500 Internal Server Error"));
      }
      return Promise.resolve(
        mockJsonResponse([{ id: "conv-repo", title: "Test conversation" }]),
      );
    });

    // Does not throw despite cloud PATCH rejection
    const result =
      await AgentServerConversationService.updateConversationRepository(
        "conv-repo",
        "OpenHands/OpenHands",
        "main",
        "github",
      );

    expect(result).toBeDefined();
    expect(result.id).toBe("conv-repo");
  });

  it("does not call cloud PATCH when active backend is local", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });

    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([
      {
        id: "conv-repo",
        created_at: "2026-10-08T00:00:00Z",
        updated_at: "2026-10-08T00:00:00Z",
      } as any,
    ]);

    await AgentServerConversationService.updateConversationRepository(
      "conv-repo",
      "OpenHands/OpenHands",
      "main",
      "github",
    );

    const patchCall = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "PATCH",
    );
    expect(patchCall).toBeUndefined();
  });
});
