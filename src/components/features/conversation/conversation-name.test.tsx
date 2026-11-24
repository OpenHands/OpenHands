import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationName } from "./conversation-name";

const updateConversationMock = vi.hoisted(() => vi.fn());

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "local" }, orgId: null }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conversation-1",
      title: "Original Title",
      execution_status: "idle",
      conversation_url: null,
      session_api_key: null,
      llm_model: "test-model",
      public: false,
      created_by_user_id: "user-1",
    },
  }),
}));

vi.mock("#/hooks/mutation/use-update-conversation", () => ({
  useUpdateConversation: () => ({ mutate: updateConversationMock }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conversation-1" }),
}));

vi.mock("#/hooks/use-conversation-display-title", () => ({
  useConversationDisplayTitleState: () => () => ({
    title: "Original Title",
    pending: false,
  }),
}));

vi.mock("#/hooks/use-conversation-name-context-menu", () => ({
  useConversationNameContextMenu: () => ({
    handleDelete: vi.fn(),
    handleStop: vi.fn(),
    handleDownloadConversation: vi.fn(),
    handleDisplayCost: vi.fn(),
    handleShowAgentTools: vi.fn(),
    handleShowSkills: vi.fn(),
    handleShowHooks: vi.fn(),
    handleTogglePublic: vi.fn(),
    handleCopyShareLink: vi.fn(),
    shareUrl: undefined,
    handleConfirmDelete: vi.fn(),
    handleConfirmStop: vi.fn(),
    systemModalVisible: false,
    setSystemModalVisible: vi.fn(),
    skillsModalVisible: false,
    setSkillsModalVisible: vi.fn(),
    hooksModalVisible: false,
    setHooksModalVisible: vi.fn(),
    confirmDeleteModalVisible: false,
    setConfirmDeleteModalVisible: vi.fn(),
    confirmStopModalVisible: false,
    setConfirmStopModalVisible: vi.fn(),
    systemMessage: null,
    shouldShowStop: false,
    shouldShowDownloadConversation: false,
    shouldShowDisplayCost: false,
    shouldShowAgentTools: false,
    shouldShowSkills: false,
    shouldShowHooks: false,
  }),
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => false,
}));

vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => false,
}));

async function openRenameInput(user: ReturnType<typeof userEvent.setup>) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ConversationName />
    </QueryClientProvider>,
  );

  await user.click(screen.getByLabelText("COMMON$MORE_OPTIONS"));
  await user.click(screen.getByTestId("rename-button"));

  const input = screen.getByTestId("conversation-name-input");
  await waitFor(() => expect(input).toHaveFocus());
  return input;
}

describe("ConversationName", () => {
  beforeEach(() => {
    updateConversationMock.mockClear();
  });

  it("cancels inline rename on Escape without saving and restores menu focus", async () => {
    const user = userEvent.setup();
    const input = await openRenameInput(user);

    await user.clear(input);
    await user.type(input, "Changed Title");
    await user.keyboard("{Escape}");

    expect(updateConversationMock).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("conversation-name-input"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-name-title")).toHaveTextContent(
      "Original Title",
    );

    const moreOptions = screen.getByLabelText("COMMON$MORE_OPTIONS");
    expect(moreOptions).toBeInTheDocument();
    await waitFor(() => expect(moreOptions).toHaveFocus());
  });

  it("keeps Enter as a save path", async () => {
    const user = userEvent.setup();
    const input = await openRenameInput(user);

    await user.clear(input);
    await user.type(input, "Saved Title");
    await user.keyboard("{Enter}");

    await waitFor(() =>
      expect(updateConversationMock).toHaveBeenCalledWith(
        { conversationId: "conversation-1", newTitle: "Saved Title" },
        expect.any(Object),
      ),
    );
    expect(
      screen.queryByTestId("conversation-name-input"),
    ).not.toBeInTheDocument();
  });

  it("keeps blur as a save path", async () => {
    const user = userEvent.setup();
    const input = await openRenameInput(user);

    await user.clear(input);
    await user.type(input, "Blurred Title");
    await user.tab();

    await waitFor(() =>
      expect(updateConversationMock).toHaveBeenCalledWith(
        { conversationId: "conversation-1", newTitle: "Blurred Title" },
        expect.any(Object),
      ),
    );
  });

  it("saves the next rename session after an Escape cancel", async () => {
    const user = userEvent.setup();
    const input = await openRenameInput(user);

    await user.clear(input);
    await user.type(input, "Canceled Title");
    await user.keyboard("{Escape}");
    expect(updateConversationMock).not.toHaveBeenCalled();

    await user.click(screen.getByLabelText("COMMON$MORE_OPTIONS"));
    await user.click(screen.getByTestId("rename-button"));
    const nextInput = screen.getByTestId("conversation-name-input");
    await waitFor(() => expect(nextInput).toHaveFocus());

    await user.clear(nextInput);
    await user.type(nextInput, "Saved After Cancel");
    await user.keyboard("{Enter}");

    await waitFor(() =>
      expect(updateConversationMock).toHaveBeenCalledWith(
        { conversationId: "conversation-1", newTitle: "Saved After Cancel" },
        expect.any(Object),
      ),
    );
    expect(
      screen.queryByTestId("conversation-name-input"),
    ).not.toBeInTheDocument();
  });
});
