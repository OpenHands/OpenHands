import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { setStoredConversationMetadata } from "#/api/conversation-metadata-store";
import {
  batchGetCloudConversations,
  createCloudAppConversation,
  searchCloudConversations,
} from "#/api/cloud/conversation-service.api";
import { AGENT_CANVAS_CLIENT_HEADERS } from "#/api/client-source";

const { mockCallCloudProxy } = vi.hoisted(() => ({
  mockCallCloudProxy: vi.fn(),
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: (...args: unknown[]) => mockCallCloudProxy(...args),
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  kind: "cloud",
  host: "https://app.all-hands.dev",
  apiKey: "secret",
  name: "Cloud",
};

describe("cloud conversation-service overlay", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    mockCallCloudProxy.mockReset();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    mockCallCloudProxy.mockReset();
  });

  it("marks Cloud conversation starts as originating from Agent Canvas", async () => {
    mockCallCloudProxy.mockResolvedValueOnce({ id: "start-task" });

    await createCloudAppConversation({
      initial_message: null,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: null,
      parent_conversation_id: null,
    });

    expect(mockCallCloudProxy).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/app-conversations",
        headers: AGENT_CANVAS_CLIENT_HEADERS,
      }),
    );
  });

  it("overlays locally-stored repo selection onto batchGetCloudConversations results when the server returns nulls", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });

    mockCallCloudProxy.mockResolvedValueOnce([
      {
        id: "conv-1",
        title: "Hello",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      },
    ]);

    const [conversation] = await batchGetCloudConversations(["conv-1"]);

    expect(conversation).not.toBeNull();
    expect(conversation?.selected_repository).toBe("octocat/hello-world");
    expect(conversation?.selected_branch).toBe("main");
    expect(conversation?.git_provider).toBe("github");
  });

  it("prefers the cloud server values over locally-stored selections when present", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "stale/local",
      selected_branch: "stale-branch",
      git_provider: "gitlab",
    });

    mockCallCloudProxy.mockResolvedValueOnce([
      {
        id: "conv-1",
        title: "Hello",
        selected_repository: "octocat/hello-world",
        selected_branch: "main",
        git_provider: "github",
      },
    ]);

    const [conversation] = await batchGetCloudConversations(["conv-1"]);

    expect(conversation?.selected_repository).toBe("octocat/hello-world");
    expect(conversation?.selected_branch).toBe("main");
    expect(conversation?.git_provider).toBe("github");
  });

  it("leaves null entries untouched when the cloud server returns null for a missing conversation", async () => {
    mockCallCloudProxy.mockResolvedValueOnce([null]);

    const result = await batchGetCloudConversations(["missing"]);

    expect(result).toEqual([null]);
  });

  it("returns an empty array without calling the proxy when no ids are provided", async () => {
    const result = await batchGetCloudConversations([]);

    expect(result).toEqual([]);
    expect(mockCallCloudProxy).not.toHaveBeenCalled();
  });

  it("only fills server-side nulls — server-populated fields are preserved", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "local/repo",
      selected_branch: "local-branch",
      git_provider: "gitlab",
    });

    mockCallCloudProxy.mockResolvedValueOnce([
      {
        id: "conv-1",
        title: "Hello",
        selected_repository: "server/repo",
        selected_branch: null,
        git_provider: null,
      },
    ]);

    const [conversation] = await batchGetCloudConversations(["conv-1"]);

    // Server-populated field wins…
    expect(conversation?.selected_repository).toBe("server/repo");
    // …but the two nulls are filled in from local storage.
    expect(conversation?.selected_branch).toBe("local-branch");
    expect(conversation?.git_provider).toBe("gitlab");
  });

  it("leaves server-side nulls null when local storage only has a partial selection", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "local/repo",
      selected_branch: null,
      git_provider: null,
    });

    mockCallCloudProxy.mockResolvedValueOnce([
      {
        id: "conv-1",
        title: "Hello",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      },
    ]);

    const [conversation] = await batchGetCloudConversations(["conv-1"]);

    expect(conversation?.selected_repository).toBe("local/repo");
    expect(conversation?.selected_branch).toBeNull();
    expect(conversation?.git_provider).toBeNull();
  });

  it("applies the overlay independently per conversation when batching multiple ids", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });
    setStoredConversationMetadata("conv-2", {
      selected_repository: "octocat/other",
      selected_branch: "dev",
      git_provider: "github",
    });

    mockCallCloudProxy.mockResolvedValueOnce([
      {
        id: "conv-1",
        title: "First",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      },
      {
        id: "conv-2",
        title: "Second",
        // Server returns the repo already — overlay should NOT override.
        selected_repository: "server/two",
        selected_branch: null,
        git_provider: null,
      },
      {
        id: "conv-3",
        title: "Third (no local data)",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      },
    ]);

    const conversations = await batchGetCloudConversations([
      "conv-1",
      "conv-2",
      "conv-3",
    ]);

    expect(conversations[0]?.selected_repository).toBe("octocat/hello-world");
    expect(conversations[0]?.selected_branch).toBe("main");
    expect(conversations[0]?.git_provider).toBe("github");

    expect(conversations[1]?.selected_repository).toBe("server/two");
    expect(conversations[1]?.selected_branch).toBe("dev");
    expect(conversations[1]?.git_provider).toBe("github");

    expect(conversations[2]?.selected_repository).toBeNull();
    expect(conversations[2]?.selected_branch).toBeNull();
    expect(conversations[2]?.git_provider).toBeNull();
  });

  it("overlays repo selection on each item returned from searchCloudConversations", async () => {
    setStoredConversationMetadata("conv-1", {
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });

    mockCallCloudProxy.mockResolvedValueOnce({
      items: [
        {
          id: "conv-1",
          title: "Hello",
          selected_repository: null,
          selected_branch: null,
          git_provider: null,
        },
        {
          id: "conv-2",
          title: "Other",
          selected_repository: null,
          selected_branch: null,
          git_provider: null,
        },
      ],
      next_page_id: null,
    });

    const page = await searchCloudConversations(20);

    expect(page.items[0].selected_repository).toBe("octocat/hello-world");
    expect(page.items[0].selected_branch).toBe("main");
    expect(page.items[0].git_provider).toBe("github");
    expect(page.items[1].selected_repository).toBeNull();
  });
});
