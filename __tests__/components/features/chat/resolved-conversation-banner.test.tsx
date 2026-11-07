import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResolvedConversationBanner } from "#/components/features/chat/resolved-conversation-banner";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { makeDefaultLocalBackend } from "#/api/backend-registry/default-backend";
import { NO_BACKEND } from "#/api/backend-registry/active-store";
import { I18nKey } from "#/i18n/declaration";
import * as toastHandlers from "#/utils/custom-toast-handlers";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

const CONVERSATION_ID = "test-conversation-id";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: CONVERSATION_ID }),
  useConversationId: () => ({ conversationId: CONVERSATION_ID }),
}));

function renderBanner(tags: Record<string, string>) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  queryClient.setQueryData(
    [
      "user",
      "conversation",
      CONVERSATION_ID,
      (makeDefaultLocalBackend() ?? NO_BACKEND).id,
      null,
    ],
    { id: CONVERSATION_ID, sandbox_status: "RUNNING", tags },
  );
  render(
    <QueryClientProvider client={queryClient}>
      <ResolvedConversationBanner />
    </QueryClientProvider>,
  );
}

describe("ResolvedConversationBanner", () => {
  let updateTagFlags: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    updateTagFlags = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTagFlags",
    );
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each(["user", "auto"])("shows for resolved=%s", (resolution) => {
    renderBanner({ resolved: resolution });

    const banner = screen.getByTestId("resolved-conversation-banner");
    expect(banner).toHaveTextContent(I18nKey.CHAT_INTERFACE$RESOLVED_TITLE);
    expect(banner).toHaveTextContent(
      I18nKey.CHAT_INTERFACE$RESOLVED_DESCRIPTION,
    );
    expect(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN,
      }),
    ).toBeEnabled();
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

  it("removes only the resolved tag when Reopen is clicked", async () => {
    const tags = { resolved: "user", project: "abc", turn: "answered" };
    updateTagFlags.mockResolvedValue({
      id: CONVERSATION_ID,
      tags: { project: "abc", turn: "answered" },
    } as unknown as AppConversation);
    renderBanner(tags);

    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN,
      }),
    );

    expect(updateTagFlags).toHaveBeenCalledWith(CONVERSATION_ID, tags, {
      resolved: false,
    });
    await waitFor(() => {
      expect(
        screen.queryByTestId("resolved-conversation-banner"),
      ).not.toBeInTheDocument();
    });
  });

  it("brings the banner back and shows an error when reopening fails", async () => {
    updateTagFlags.mockRejectedValue(new Error("boom"));
    // The refetch after the failure still sees the server's resolved tag.
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([
      {
        id: CONVERSATION_ID,
        sandbox_status: "RUNNING",
        tags: { resolved: "auto" },
      } as unknown as AppConversation,
    ]);
    const toast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    renderBanner({ resolved: "auto" });

    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN,
      }),
    );

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED,
      );
    });
    expect(
      await screen.findByTestId("resolved-conversation-banner"),
    ).toBeInTheDocument();
  });
});
