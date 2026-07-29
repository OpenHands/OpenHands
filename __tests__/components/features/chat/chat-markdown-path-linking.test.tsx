import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatMessage } from "#/components/features/chat/chat-message";
import { useFilesTabStore } from "#/stores/files-tab-store";

const openWorkspaceFile = vi.fn();

vi.mock("#/services/canvas-ui", () => ({
  openWorkspaceFile: (...args: unknown[]) => openWorkspaceFile(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

vi.mock("#/hooks/query/use-workspace-files", () => ({
  useWorkspaceFiles: () => ({
    data: ["test.md", "motivational_message.md", "src/app.ts"],
    isLoading: false,
  }),
}));

describe("assistant chat Markdown path linking", () => {
  beforeEach(() => {
    openWorkspaceFile.mockClear();
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      contentViewNonce: 0,
    });
  });

  it("opens the Files drawer when an existing workspace path is clicked", async () => {
    const user = userEvent.setup();

    render(<ChatMessage type="agent" message={"Created `test.md`"} />);

    await user.click(screen.getByTestId("markdown-file-path-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith("test.md", "conv-1");
  });

  it("opens the Files drawer for bold-emphasized existing paths", async () => {
    const user = userEvent.setup();

    render(
      <ChatMessage
        type="agent"
        message={
          "The file **motivational_message.md** has been created with an inspiring quote."
        }
      />,
    );

    await user.click(screen.getByTestId("markdown-file-path-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith(
      "motivational_message.md",
      "conv-1",
    );
  });

  it("does not link paths that are not in the workspace", () => {
    render(
      <ChatMessage type="agent" message={"Send me exactly same text - profile.md"} />,
    );

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/profile\.md/)).toBeInTheDocument();
  });

  it("does not link missing backtick paths either", () => {
    render(<ChatMessage type="agent" message={"See `profile.md` please"} />);

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("profile.md").tagName).toBe("CODE");
  });

  it("keeps path code nested in a Markdown link as plain code", () => {
    render(
      <ChatMessage
        type="agent"
        message={"See [`src/app.ts`](https://example.com)"}
      />,
    );

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("src/app.ts").tagName).toBe("CODE");
  });

  it("does not link ordinary dotted / non-path inline code", () => {
    render(
      <ChatMessage type="agent" message={"Use `console.log` and `v1.2.3`"} />,
    );

    expect(
      screen.queryByTestId("markdown-file-path-link"),
    ).not.toBeInTheDocument();
  });
});
