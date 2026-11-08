import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMemoryRouter,
  matchRoutes,
  RouterProvider,
  Outlet,
  type RouteObject,
  useNavigate,
} from "react-router";
import routeConfig from "../routes";
import * as automationInterface from "#/manifests/automation-interface";
import { createInstance } from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { translationResources } from "#/i18n/resources";
import { useAppTitle } from "./use-app-title";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

vi.unmock("react-i18next");

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

function makeConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: "123",
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "My Conversation",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2026-10-18T00:00:00Z",
    updated_at: "2026-10-18T00:00:00Z",
    execution_status: null,
    sandbox_status: null,
    conversation_url: null,
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  };
}

const routeModules = import.meta.glob<{ handle?: unknown }>("../routes/*.tsx");

// Use the real route registrations and their exported handles, without mounting
// each page's unrelated UI or loaders. A catch-all therefore really is a 404.
function titleRoutes(
  entries: typeof routeConfig,
  probe: React.ReactNode,
): RouteObject[] {
  return entries.map((entry) => {
    const common = {
      id: entry.file,
      HydrateFallback: () => null,
      lazy: async () => ({
        handle: (await routeModules[`../${entry.file}`]()).handle,
      }),
      element:
        entry.file === "routes/root-layout.tsx" ? (
          <>
            {probe}
            <Outlet />
          </>
        ) : (
          <Outlet />
        ),
    };
    return entry.index
      ? { ...common, index: true }
      : {
          ...common,
          path: entry.path,
          children: entry.children
            ? titleRoutes(entry.children, probe)
            : undefined,
        };
  });
}

async function renderAppTitleHook(path = "/", basename?: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const i18n = createInstance();
  void i18n.use(initReactI18next).init({
    lng: "en",
    fallbackLng: "en",
    defaultNS: "openhands",
    initAsync: false,
    resources: Object.fromEntries(
      Object.entries(translationResources).map(([language, translations]) => [
        language,
        { openhands: translations },
      ]),
    ),
  });
  // Await real route-module imports before rendering so their first-load cost
  // is not mistaken for a failed title update by Testing Library's 1s poll.
  await Promise.all(
    (matchRoutes(titleRoutes(routeConfig, null), path, basename) ?? []).map(
      ({ route }) => routeModules[`../${route.id}`](),
    ),
  );
  let router: ReturnType<typeof createMemoryRouter>;
  const hook = renderHook(
    () => ({ title: useAppTitle(), navigate: useNavigate() }),
    {
      wrapper: ({ children }) => {
        router ??= createMemoryRouter(titleRoutes(routeConfig, children), {
          basename,
          initialEntries: [path],
        });
        return (
          <QueryClientProvider client={queryClient}>
            <I18nextProvider i18n={i18n}>
              <RouterProvider router={router} />
            </I18nextProvider>
          </QueryClientProvider>
        );
      },
    },
  );
  await waitFor(() => expect(hook.result.current).not.toBeNull());
  return { ...hook, queryClient, i18n };
}

describe("useAppTitle", () => {
  const getConversations = vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  );

  beforeEach(() => {
    vi.restoreAllMocks();
    getConversations.mockReset().mockResolvedValue([makeConversation()]);
    useConversationStateStore.getState().reset();
  });

  // @spec AWI-001 — Localized section titles
  it.each([
    ["/", "New Chat | OpenHands"],
    ["/conversations", "New Chat | OpenHands"],
    ["/settings", "Settings | OpenHands"],
    ["/settings/llm?tab=profiles", "Settings | OpenHands"],
    ["/customize", "Customize | OpenHands"],
    ["/mcp", "Model Context Protocol (MCP) | OpenHands"],
    ["/skills", "Skills | OpenHands"],
    ["/plugins", "Plugins | OpenHands"],
    ["/apps", "Apps for Agent Canvas | OpenHands"],
    ["/settings-other", "OpenHands"],
    ["/settings/does-not-exist", "OpenHands"],
    ["/settings/llm/deeper", "OpenHands"],
    ["/automations/a/b", "OpenHands"],
  ])("identifies the section at %s", async (path, expected) => {
    const { result } = await renderAppTitleHook(path);
    expect(result.current.title).toBe(expected);
    expect(getConversations).not.toHaveBeenCalled();
  });

  it("uses the admitted automation label and falls back when the interface is absent", async () => {
    const copy = automationInterface.getInterfaceCopy();
    const getCopy = vi
      .spyOn(automationInterface, "getInterfaceCopy")
      .mockReturnValue({
        ...copy,
        sidebarLabel: "Team workflows",
      });
    const gate = vi
      .spyOn(automationInterface, "hasAutomationInterface")
      .mockReturnValue(true);
    const { result } = await renderAppTitleHook("/automations/templates");
    expect(result.current.title).toBe("Team workflows | OpenHands");

    gate.mockReturnValue(false);
    getCopy.mockImplementation(() => {
      throw new Error("Unavailable interface must not be read");
    });
    await act(() => result.current.navigate("/automations"));
    expect(result.current.title).toBe("OpenHands");
  });

  it("updates section titles on navigation and language changes under a base path", async () => {
    const { result, i18n } = await renderAppTitleHook("/canvas/", "/canvas");
    expect(result.current.title).toBe("New Chat | OpenHands");

    await act(() => result.current.navigate("/settings/llm"));
    expect(result.current.title).toBe("Settings | OpenHands");

    await act(() => i18n.changeLanguage("de"));
    expect(result.current.title).toBe("Einstellungen | OpenHands");
  });

  it.each(["/conversations/123", "/conversations/123/panel"])(
    "uses the conversation title at %s and updates it after refetch",
    async (path) => {
      const { result, queryClient } = await renderAppTitleHook(path);
      await waitFor(() =>
        expect(result.current.title).toBe("My Conversation | OpenHands"),
      );

      getConversations.mockResolvedValue([
        makeConversation({ title: "Renamed conversation" }),
      ]);
      await act(() => queryClient.invalidateQueries());

      await waitFor(() =>
        expect(result.current.title).toBe("Renamed conversation | OpenHands"),
      );
    },
  );

  // @spec AWI-001 — Stable conversation titles and status
  it("keeps a stable title while loading and when the conversation is untitled", async () => {
    let resolveConversation!: (value: AppConversation[]) => void;
    getConversations.mockReturnValue(
      new Promise((resolve) => {
        resolveConversation = resolve;
      }),
    );
    const { result, queryClient } =
      await renderAppTitleHook("/conversations/123");
    expect(result.current.title).toBe("OpenHands");

    await act(() => resolveConversation([makeConversation({ title: null })]));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(result.current.title).toBe("OpenHands");
  });

  it.each([
    [ExecutionStatus.RUNNING, "🟢"],
    [ExecutionStatus.FINISHED, "✅"],
    [ExecutionStatus.IDLE, "✅"],
    [ExecutionStatus.WAITING_FOR_CONFIRMATION, "✅"],
    [ExecutionStatus.PAUSED, "⚪"],
    [ExecutionStatus.ERROR, "🔴"],
    [ExecutionStatus.STUCK, "🔴"],
  ])("preserves the %s status emoji", async (status, emoji) => {
    const { result } = await renderAppTitleHook("/conversations/123");
    await waitFor(() =>
      expect(result.current.title).toBe("My Conversation | OpenHands"),
    );

    act(() =>
      useConversationStateStore.getState().setExecutionStatus("123", status),
    );

    expect(result.current.title).toBe(`${emoji} My Conversation | OpenHands`);
  });

  it("uses the saved status until a live status arrives", async () => {
    getConversations.mockResolvedValue([
      makeConversation({ execution_status: ExecutionStatus.RUNNING }),
    ]);
    const { result } = await renderAppTitleHook("/conversations/123");
    await waitFor(() =>
      expect(result.current.title).toBe("🟢 My Conversation | OpenHands"),
    );

    act(() =>
      useConversationStateStore
        .getState()
        .setExecutionStatus("123", ExecutionStatus.ERROR),
    );

    expect(result.current.title).toBe("🔴 My Conversation | OpenHands");
  });
});
