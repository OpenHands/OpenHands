import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GitToolsSubmenu } from "./git-tools-submenu";

const mocks = vi.hoisted(() => ({
  setMessageToSend: vi.fn(),
  useActiveConversation: vi.fn(),
  useTranslation: vi.fn(() => ({ t: (key: string) => key })),
  getGitPullPrompt: vi.fn(() => "pull prompt"),
  getGitPushPrompt: vi.fn(() => "push prompt"),
  getCreatePRPrompt: vi.fn(() => "pr prompt"),
  getCreateNewBranchPrompt: vi.fn(() => "branch prompt"),
}));

vi.mock("react-i18next", () => ({
  useTranslation: mocks.useTranslation,
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({ setMessageToSend: mocks.setMessageToSend }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: mocks.useActiveConversation,
}));

vi.mock("#/utils/utils", () => ({
  getGitPullPrompt: mocks.getGitPullPrompt,
  getGitPushPrompt: mocks.getGitPushPrompt,
  getCreatePRPrompt: mocks.getCreatePRPrompt,
  getCreateNewBranchPrompt: mocks.getCreateNewBranchPrompt,
}));

vi.mock("#/ui/context-menu", () => ({
  ContextMenu: ({
    children,
    testId,
    className,
  }: {
    children: React.ReactNode;
    testId: string;
    className: string;
  }) => (
    <div data-testid={testId} className={className}>
      {children}
    </div>
  ),
}));

vi.mock("../context-menu/context-menu-list-item", () => ({
  ContextMenuListItem: ({
    children,
    testId,
    onClick,
    className,
  }: {
    children: React.ReactNode;
    testId: string;
    onClick: () => void;
    className: string;
  }) => (
    <button
      type="button"
      data-testid={testId}
      className={className}
      onClick={onClick}
    >
      {children}
    </button>
  ),
}));

vi.mock("./tools-context-menu-icon-text", () => ({
  ToolsContextMenuIconText: ({ text }: { text: string }) => <span>{text}</span>,
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.useActiveConversation.mockReturnValue({
    data: { git_provider: "github" },
  });
});

describe("GitToolsSubmenu", () => {
  it("renders every git action in the submenu", async () => {
    vi.resetModules();
    const { GitToolsSubmenu: ReloadedGitToolsSubmenu } =
      await import("./git-tools-submenu");
    render(<ReloadedGitToolsSubmenu onClose={vi.fn()} />);

    expect(mocks.useTranslation).toHaveBeenCalledWith("openhands");
    expect(screen.getByTestId("git-tools-submenu")).toHaveClass("w-max");
    expect(screen.getByTestId("git-pull-button")).toHaveTextContent(
      "COMMON$GIT_PULL",
    );
    expect(screen.getByTestId("git-push-button")).toHaveTextContent(
      "COMMON$GIT_PUSH",
    );
    expect(screen.getByTestId("create-pr-button")).toHaveTextContent(
      "COMMON$CREATE_PR",
    );
    expect(screen.getByTestId("create-new-branch-button")).toHaveTextContent(
      "COMMON$CREATE_NEW_BRANCH",
    );
    for (const button of screen.getAllByRole("button")) {
      expect(button).toHaveClass("!w-auto", "whitespace-nowrap");
    }
  });

  it.each([
    ["git-pull-button", mocks.getGitPullPrompt, "pull prompt"],
    ["git-push-button", mocks.getGitPushPrompt, "push prompt"],
    ["create-pr-button", mocks.getCreatePRPrompt, "pr prompt"],
    [
      "create-new-branch-button",
      mocks.getCreateNewBranchPrompt,
      "branch prompt",
    ],
  ])(
    "builds and queues the %s action before closing",
    (testId, prompt, text) => {
      const onClose = vi.fn();
      render(<GitToolsSubmenu onClose={onClose} />);

      fireEvent.click(screen.getByTestId(testId));

      expect(prompt).toHaveBeenCalledTimes(1);
      expect(mocks.setMessageToSend).toHaveBeenCalledWith(text);
      expect(mocks.setMessageToSend.mock.invocationCallOrder[0]).toBeLessThan(
        onClose.mock.invocationCallOrder[0],
      );
      expect(onClose).toHaveBeenCalledTimes(1);
    },
  );

  it("passes the active provider to provider-specific prompts", () => {
    render(<GitToolsSubmenu onClose={vi.fn()} />);

    fireEvent.click(screen.getByTestId("git-push-button"));
    fireEvent.click(screen.getByTestId("create-pr-button"));

    expect(mocks.getGitPushPrompt).toHaveBeenCalledWith("github");
    expect(mocks.getCreatePRPrompt).toHaveBeenCalledWith("github");
  });

  it("uses the provider-agnostic prompt fallback without an active conversation", () => {
    mocks.useActiveConversation.mockReturnValue({ data: undefined });
    render(<GitToolsSubmenu onClose={vi.fn()} />);

    fireEvent.click(screen.getByTestId("git-push-button"));
    fireEvent.click(screen.getByTestId("create-pr-button"));

    expect(mocks.getGitPushPrompt).toHaveBeenCalledWith(undefined);
    expect(mocks.getCreatePRPrompt).toHaveBeenCalledWith(undefined);
  });
});
