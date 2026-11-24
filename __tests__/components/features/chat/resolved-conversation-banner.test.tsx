import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResolvedConversationBanner } from "#/components/features/chat/resolved-conversation-banner";
import { HttpError } from "@openhands/typescript-client";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { makeDefaultLocalBackend } from "#/api/backend-registry/default-backend";
import {
  NO_BACKEND,
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useEventStore } from "#/stores/use-event-store";
import { I18nKey } from "#/i18n/declaration";
import * as toastHandlers from "#/utils/custom-toast-handlers";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { MessageEvent } from "#/types/agent-server/core";

const projectsFlag = vi.hoisted(() => ({ enabled: true }));

vi.mock("#/hooks/query/use-projects", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/hooks/query/use-projects")>()),
  useProjectsEnabled: () => projectsFlag.enabled,
}));

const CONVERSATION_ID = "test-conversation-id";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: CONVERSATION_ID }),
  useConversationId: () => ({ conversationId: CONVERSATION_ID }),
}));

function conversationWith(tags: Record<string, string>) {
  return {
    id: CONVERSATION_ID,
    sandbox_status: "RUNNING",
    tags,
  } as unknown as AppConversation;
}

function detailKey(backendId: string) {
  return ["user", "conversation", CONVERSATION_ID, backendId, null];
}

const localBackendId = (makeDefaultLocalBackend() ?? NO_BACKEND).id;

function renderBanner(
  tags: Record<string, string>,
  backendId = localBackendId,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  // Only a project's members can be resolved; tests opt out with project: "".
  const withProject = { project: "abc", ...tags };
  if (!withProject.project)
    delete (withProject as Record<string, string>).project;
  queryClient.setQueryData(detailKey(backendId), conversationWith(withProject));
  render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <div>
          <ResolvedConversationBanner />
          <div contentEditable data-testid="composer" />
        </div>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return queryClient;
}

function userMessage(id: string): MessageEvent {
  return {
    id,
    timestamp: new Date().toISOString(),
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
    activated_skills: [],
    extended_content: [],
  } as MessageEvent;
}

const reopenButton = () =>
  screen.getByRole("button", { name: I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN });

describe("ResolvedConversationBanner", () => {
  let batchGet: ReturnType<typeof vi.spyOn>;
  let setResolved: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    setResolved = vi
      .spyOn(ProjectsService, "setConversationResolved")
      .mockResolvedValue(undefined);
    useEventStore.setState({ events: [], eventIds: new Set(), uiEvents: [] });
    batchGet = vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    __resetActiveStoreForTests();
    projectsFlag.enabled = true;
  });

  it("stays hidden while Projects is off", () => {
    projectsFlag.enabled = false;
    renderBanner({ resolved: "user" });

    expect(screen.queryByTestId("resolved-conversation-banner")).toBeNull();
  });

  it.each(["user", "auto"])("shows for resolved=%s", (resolution) => {
    renderBanner({ resolved: resolution });

    const banner = screen.getByTestId("resolved-conversation-banner");
    expect(banner).toHaveTextContent(I18nKey.CHAT_INTERFACE$RESOLVED_TITLE);
    expect(banner).toHaveTextContent(
      I18nKey.CHAT_INTERFACE$RESOLVED_DESCRIPTION,
    );
    expect(screen.getByRole("status")).not.toContainElement(reopenButton());
    expect(reopenButton()).toBeEnabled();
  });

  it.each([
    ["0.93 nothing", I18nKey.PROJECTS$AUTO_RESOLVED],
    ["agent", I18nKey.PROJECTS$AUTO_RESOLVED_AGENT],
  ])("says why it resolved itself (%s)", (resolvenote, message) => {
    renderBanner({ resolved: "auto", resolvenote });

    expect(
      screen.getByTestId("resolved-conversation-auto-note"),
    ).toHaveTextContent(message);
  });

  it.each<Record<string, string>>([
    { resolved: "auto" },
    { resolved: "auto", resolvenote: "sure" },
    { resolved: "user", resolvenote: "0.93 nothing" },
  ])("gives no reason for tags %j", (tags) => {
    renderBanner(tags);

    expect(
      screen.queryByTestId("resolved-conversation-auto-note"),
    ).not.toBeInTheDocument();
  });

  it.each<Record<string, string>>([
    {},
    { resolved: "" },
    { resolved: "true" },
    { project: "abc" },
    { resolved: "user", project: "" },
  ])("stays hidden for tags %j", (tags) => {
    renderBanner(tags);

    expect(
      screen.queryByTestId("resolved-conversation-banner"),
    ).not.toBeInTheDocument();
  });

  it("stays hidden on a cloud backend", () => {
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "key",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });

    renderBanner({ resolved: "user" }, "cloud-1");

    expect(
      screen.queryByTestId("resolved-conversation-banner"),
    ).not.toBeInTheDocument();
  });

  it("reopens a project's member through the project's route", async () => {
    batchGet.mockResolvedValue([conversationWith({ project: "abc" })]);
    renderBanner({
      resolved: "auto",
      project: "abc",
      resolvenote: "0.9 critic",
    });

    await userEvent.click(reopenButton());

    expect(setResolved).toHaveBeenCalledWith("abc", CONVERSATION_ID, false);
    await waitFor(() => {
      expect(
        screen.queryByTestId("resolved-conversation-banner"),
      ).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("composer")).toHaveFocus();
  });

  it("shows an error when the project's route refuses the reopen", async () => {
    setResolved.mockRejectedValue(new HttpError(500, "Server Error", {}));
    const toast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    renderBanner({ resolved: "user", project: "abc" });

    await userEvent.click(reopenButton());

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED,
      ),
    );
    expect(
      screen.getByTestId("resolved-conversation-banner"),
    ).toBeInTheDocument();
  });

  it("refetches the conversation when a new user message is echoed", async () => {
    batchGet.mockResolvedValue([conversationWith({ project: "abc" })]);
    renderBanner({ resolved: "auto", project: "abc" });
    expect(batchGet).not.toHaveBeenCalled();

    act(() => {
      useEventStore.getState().addEvent(userMessage("user-1"));
    });

    await waitFor(() => {
      expect(
        screen.queryByTestId("resolved-conversation-banner"),
      ).not.toBeInTheDocument();
    });
    expect(batchGet).toHaveBeenCalledWith([CONVERSATION_ID]);
  });

  it("keeps the banner when the server still reports it resolved after a message", async () => {
    batchGet.mockResolvedValue([conversationWith({ resolved: "auto" })]);
    renderBanner({ resolved: "auto" });

    act(() => {
      useEventStore.getState().addEvent(userMessage("user-1"));
    });

    await waitFor(() => expect(batchGet).toHaveBeenCalled());
    expect(
      screen.getByTestId("resolved-conversation-banner"),
    ).toBeInTheDocument();
  });
});
