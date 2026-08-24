import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { ChatMessage } from "#/components/features/chat/chat-message";
import { WorkspaceFilesForChatContext } from "#/components/features/chat/chat-markdown-path-code";
import { useFilesTabStore } from "#/stores/files-tab-store";

const openWorkspaceFile = vi.fn();

vi.mock("#/services/canvas-ui", () => ({
  openWorkspaceFile: (...args: unknown[]) => openWorkspaceFile(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

const WORKSPACE_FILES = ["test.md", "motivational_message.md", "src/app.ts"];

function renderAgentMessage(
  message: string,
  files: string[] = WORKSPACE_FILES,
) {
  return render(
    <WorkspaceFilesForChatContext.Provider value={files}>
      <ChatMessage type="agent" message={message} />
    </WorkspaceFilesForChatContext.Provider>,
  );
}

describe("assistant chat Markdown path linking", () => {
  beforeEach(() => {
    openWorkspaceFile.mockClear();
    ConversationService.setCurrentConversation(null);
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      openPaths: [],
    });
  });

  it("opens the Files drawer when an existing workspace path is clicked", async () => {
    const user = userEvent.setup();

    renderAgentMessage("Created `test.md`");

    await user.click(screen.getByTestId("markdown-file-path-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith("test.md", "conv-1");
  });

  it("opens the Files drawer for bold-emphasized existing paths", async () => {
    const user = userEvent.setup();

    renderAgentMessage(
      "The file **motivational_message.md** has been created with an inspiring quote.",
    );

    await user.click(screen.getByTestId("markdown-file-path-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith(
      "motivational_message.md",
      "conv-1",
    );
  });

  it("links absolute nested working-dir paths using ConversationService working_dir", async () => {
    const user = userEvent.setup();
    ConversationService.setCurrentConversation({
      id: "conv-1",
      workspace: { working_dir: "/workspace/project/packages/app" },
    } as never);

    renderAgentMessage("See `/workspace/project/packages/app/src/index.ts`", [
      "src/index.ts",
    ]);

    await user.click(screen.getByTestId("markdown-file-path-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith("src/index.ts", "conv-1");
  });

  it("does not link paths that are not in the workspace", () => {
    renderAgentMessage("Send me exactly same text - profile.md");

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/profile\.md/)).toBeInTheDocument();
  });

  it("does not link missing backtick paths either", () => {
    renderAgentMessage("See `profile.md` please");

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("profile.md").tagName).toBe("CODE");
  });

  it("keeps path code nested in a Markdown link as plain code", () => {
    renderAgentMessage("See [`src/app.ts`](https://example.com)");

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("src/app.ts").tagName).toBe("CODE");
  });

  it("does not link ordinary dotted / non-path inline code", () => {
    renderAgentMessage("Use `console.log` and `v1.2.3`");

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
  });
});
