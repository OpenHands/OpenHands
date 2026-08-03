import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SkillsService from "#/api/skills-service";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";

const state = vi.hoisted(() => ({
  backendId: "backend-1",
  backendKind: "local" as "local" | "cloud",
  orgId: null as string | null,
  routeConversationId: null as string | null,
  conversation: undefined as
    | {
        id: string;
        conversation_url: string | null;
        sandbox_status: string;
        selected_workspace: string | null;
      }
    | undefined,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: state.backendId, kind: state.backendKind },
    orgId: state.orgId,
  }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({
    conversationId: state.routeConversationId,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: state.conversation }),
}));

function makeWrapper(
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  }),
) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe("useConversationSkills", () => {
  beforeEach(() => {
    state.backendKind = "local";
    state.backendId = "backend-1";
    state.orgId = null;
    state.routeConversationId = null;
    state.conversation = undefined;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads the workspace catalog for local conversations", async () => {
    state.routeConversationId = "conversation-1";
    state.conversation = {
      id: "conversation-1",
      conversation_url:
        "http://localhost:8000/api/conversations/conversation-1",
      sandbox_status: "RUNNING",
      selected_workspace: "/workspace/project",
    };
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValue([]);
    const conversationSpy = vi.spyOn(
      SkillsService,
      "getConversationLoadedSkills",
    );

    renderHook(() => useConversationSkills(), { wrapper: makeWrapper() });

    await waitFor(() =>
      expect(catalogSpy).toHaveBeenCalledWith("/workspace/project"),
    );
    expect(conversationSpy).not.toHaveBeenCalled();
  });

  it("keeps using the available catalog on active Cloud conversations", async () => {
    state.backendKind = "cloud";
    state.routeConversationId = "conversation-1";
    state.conversation = {
      id: "conversation-1",
      conversation_url:
        "https://runtime.example/api/conversations/conversation-1",
      sandbox_status: "RUNNING",
      selected_workspace: "/workspace/project",
    };
    const catalogSpy = vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
      {
        name: "custom-skill",
        type: "agentskills",
        source: "project",
        triggers: ["/custom"],
      },
    ]);
    const conversationSpy = vi.spyOn(
      SkillsService,
      "getConversationLoadedSkills",
    );

    const { result } = renderHook(() => useConversationSkills(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.data).toHaveLength(1));
    expect(catalogSpy).toHaveBeenCalledWith("/workspace/project");
    expect(conversationSpy).not.toHaveBeenCalled();
  });

  it("uses the available-skills catalog on the Cloud home page", async () => {
    state.backendKind = "cloud";
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValue([]);
    const conversationSpy = vi.spyOn(
      SkillsService,
      "getConversationLoadedSkills",
    );

    renderHook(() => useConversationSkills(), { wrapper: makeWrapper() });

    await waitFor(() => expect(catalogSpy).toHaveBeenCalledWith(undefined));
    expect(conversationSpy).not.toHaveBeenCalled();
  });

  it("does not substitute the global catalog while conversation metadata loads", async () => {
    state.routeConversationId = "conversation-1";
    state.conversation = undefined;
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    queryClient.setQueryData(
      ["skills", "backend-1", null, "workspace", null],
      [
        {
          name: "cached-home-skill",
          type: "agentskills",
          source: "user",
          triggers: ["/cached-home-skill"],
        },
      ],
    );
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValue([]);

    const { result } = renderHook(() => useConversationSkills(), {
      wrapper: makeWrapper(queryClient),
    });

    await Promise.resolve();
    expect(result.current.fetchStatus).toBe("idle");
    expect(result.current.data).toBeUndefined();
    expect(catalogSpy).not.toHaveBeenCalled();
  });

  it("does not query a catalog for a provisioning task route", async () => {
    state.routeConversationId = "task-abc";
    state.conversation = undefined;
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValue([]);

    const { result } = renderHook(() => useConversationSkills(), {
      wrapper: makeWrapper(),
    });

    await Promise.resolve();
    expect(result.current.fetchStatus).toBe("idle");
    expect(catalogSpy).not.toHaveBeenCalled();
  });

  it("fetches a distinct available catalog when the Cloud organization changes", async () => {
    state.backendKind = "cloud";
    state.orgId = "org-a";
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValueOnce([
        {
          name: "org-a-skill",
          type: "agentskills",
          source: "user",
          triggers: ["/org-a-skill"],
        },
      ])
      .mockResolvedValueOnce([
        {
          name: "org-b-skill",
          type: "agentskills",
          source: "user",
          triggers: ["/org-b-skill"],
        },
      ]);

    const { result, rerender } = renderHook(() => useConversationSkills(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data?.[0]?.name).toBe("org-a-skill"),
    );

    state.orgId = "org-b";
    rerender();

    await waitFor(() =>
      expect(result.current.data?.[0]?.name).toBe("org-b-skill"),
    );
    expect(catalogSpy).toHaveBeenCalledTimes(2);
    expect(catalogSpy).toHaveBeenNthCalledWith(1, undefined);
    expect(catalogSpy).toHaveBeenNthCalledWith(2, undefined);
  });

  it("fetches a distinct available catalog when the local workspace changes", async () => {
    state.routeConversationId = "conversation-1";
    state.conversation = {
      id: "conversation-1",
      conversation_url:
        "http://localhost:8000/api/conversations/conversation-1",
      sandbox_status: "RUNNING",
      selected_workspace: "/workspace/one",
    };
    const catalogSpy = vi
      .spyOn(SkillsService, "getSkills")
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    const { rerender } = renderHook(() => useConversationSkills(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() =>
      expect(catalogSpy).toHaveBeenCalledWith("/workspace/one"),
    );

    state.conversation = {
      ...state.conversation,
      selected_workspace: "/workspace/two",
    };
    rerender();

    await waitFor(() =>
      expect(catalogSpy).toHaveBeenCalledWith("/workspace/two"),
    );
    expect(catalogSpy).toHaveBeenCalledTimes(2);
  });
});
