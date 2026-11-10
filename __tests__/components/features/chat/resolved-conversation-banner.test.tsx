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

const CONVERSATION_ID = "test-conversation-id";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: CONVERSATION_ID }),
  useConversationId: () => ({ conversationId: CONVERSATION_ID }),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

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
  queryClient.setQueryData(detailKey(backendId), conversationWith(tags));
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
  let updateTagFlags: ReturnType<typeof vi.spyOn>;
  let batchGet: ReturnType<typeof vi.spyOn>;
  let setResolved: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // A server without the project's reopen route, unless a test says so.
    setResolved = vi
      .spyOn(ProjectsService, "setConversationResolved")
      .mockRejectedValue(
        new HttpError(404, "Not Found", { detail: "Not Found" }),
      );
    useEventStore.setState({ events: [], eventIds: new Set(), uiEvents: [] });
    updateTagFlags = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTagFlags",
    );
    batchGet = vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    __resetActiveStoreForTests();
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

  it("says why the server resolved it", () => {
    renderBanner({ resolved: "auto", resolvenote: "0.93 nothing" });

    expect(
      screen.getByTestId("resolved-conversation-auto-note"),
    ).toHaveTextContent(I18nKey.PROJECTS$AUTO_RESOLVED);
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

  it("reopens from fresh tags, hides before the PATCH settles, and focuses the composer", async () => {
    // The server added a tag since the cached copy was fetched.
    batchGet.mockResolvedValueOnce([
      conversationWith({ resolved: "user", project: "abc", turn: "answered" }),
    ]);
    const patch = deferred<AppConversation>();
    updateTagFlags.mockReturnValue(patch.promise);
    const queryClient = renderBanner({ resolved: "user", project: "abc" });

    await userEvent.click(reopenButton());

    expect(updateTagFlags).toHaveBeenCalledWith(
      CONVERSATION_ID,
      { resolved: "user", project: "abc", turn: "answered" },
      { resolved: false, resolvenote: false },
    );
    await waitFor(() => {
      expect(
        screen.queryByTestId("resolved-conversation-banner"),
      ).not.toBeInTheDocument();
    });
    expect(
      queryClient.getQueryData<AppConversation>(detailKey(localBackendId))
        ?.tags,
    ).toEqual({ project: "abc", turn: "answered" });
    expect(screen.getByTestId("composer")).toHaveFocus();

    batchGet.mockReturnValue(new Promise(() => {}));
    patch.resolve(conversationWith({ project: "abc", turn: "answered" }));
  });

  it("reopens a project's member through the project's route", async () => {
    setResolved.mockResolvedValue(undefined);
    batchGet.mockResolvedValue([conversationWith({ project: "abc" })]);
    renderBanner({
      resolved: "auto",
      project: "abc",
      resolvenote: "0.9 critic",
    });

    await userEvent.click(reopenButton());

    expect(setResolved).toHaveBeenCalledWith("abc", CONVERSATION_ID, false);
    expect(updateTagFlags).not.toHaveBeenCalled();
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
    expect(updateTagFlags).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("resolved-conversation-banner"),
    ).toBeInTheDocument();
  });

  it("rolls the banner back and shows an error when reopening fails", async () => {
    // Keep every refetch pending so only the rollback can restore the banner.
    batchGet.mockResolvedValueOnce([conversationWith({ resolved: "auto" })]);
    const patch = deferred<AppConversation>();
    updateTagFlags.mockReturnValue(patch.promise);
    const toast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    renderBanner({ resolved: "auto" });

    await userEvent.click(reopenButton());
    await waitFor(() => {
      expect(
        screen.queryByTestId("resolved-conversation-banner"),
      ).not.toBeInTheDocument();
    });

    batchGet.mockReturnValue(new Promise(() => {}));
    await act(async () => patch.reject(new Error("boom")));

    expect(
      await screen.findByTestId("resolved-conversation-banner"),
    ).toBeInTheDocument();
    expect(toast).toHaveBeenCalledWith(
      I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED,
    );
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
