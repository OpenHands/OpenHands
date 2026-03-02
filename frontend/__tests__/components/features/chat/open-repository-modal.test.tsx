import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { OpenRepositoryModal } from "#/components/features/chat/open-repository-modal";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

// Mock useUserProviders - default to single provider (no dropdown shown)
const mockProviders = vi.hoisted(() => ({
  current: ["github"] as string[],
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: mockProviders.current,
    isLoadingSettings: false,
  }),
}));

// Mock GitProviderDropdown
vi.mock(
  "#/components/features/home/git-provider-dropdown/git-provider-dropdown",
  () => ({
    GitProviderDropdown: ({
      providers,
      onChange,
    }: {
      providers: string[];
      onChange: (provider: string | null) => void;
    }) => (
      <div data-testid="git-provider-dropdown">
        {providers.map((p: string) => (
          <button
            key={p}
            data-testid={`provider-${p}`}
            onClick={() => onChange(p)}
          >
            {p}
          </button>
        ))}
      </div>
    ),
  }),
);

// Mock GitRepoDropdown
vi.mock(
  "#/components/features/home/git-repo-dropdown/git-repo-dropdown",
  () => ({
    GitRepoDropdown: ({
      onChange,
    }: {
      onChange: (repo?: {
        id: number;
        full_name: string;
        git_provider: string;
        main_branch: string;
      }) => void;
    }) => (
      <button
        data-testid="git-repo-dropdown"
        onClick={() =>
          onChange({
            id: 1,
            full_name: "owner/repo",
            git_provider: "github",
            main_branch: "main",
          })
        }
      >
        Mock Repo Dropdown
      </button>
    ),
  }),
);

// Mock GitBranchDropdown
vi.mock(
  "#/components/features/home/git-branch-dropdown/git-branch-dropdown",
  () => ({
    GitBranchDropdown: ({
      onBranchSelect,
      disabled,
    }: {
      onBranchSelect: (branch: { name: string } | null) => void;
      disabled: boolean;
    }) => (
      <button
        data-testid="git-branch-dropdown"
        disabled={disabled}
        onClick={() => onBranchSelect({ name: "main" })}
      >
        Mock Branch Dropdown
      </button>
    ),
  }),
);

// Mock RepoForkedIcon
vi.mock("#/icons/repo-forked.svg?react", () => ({
  default: () => <div data-testid="repo-forked-icon" />,
}));

describe("OpenRepositoryModal", () => {
  const mockOnClose = vi.fn();
  const mockOnLaunch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockProviders.current = ["github"];
  });

  it("should not render when isOpen is false", () => {
    render(
      <OpenRepositoryModal
        isOpen={false}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    expect(
      screen.queryByText("CONVERSATION$OPEN_REPOSITORY"),
    ).not.toBeInTheDocument();
  });

  it("should render modal with title and description when open", () => {
    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    expect(
      screen.getByText("CONVERSATION$OPEN_REPOSITORY"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION$SELECT_OR_INSERT_LINK"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("repo-forked-icon")).toBeInTheDocument();
  });

  it("should render Launch and Cancel buttons", () => {
    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    expect(screen.getByText("BUTTON$LAUNCH")).toBeInTheDocument();
    expect(screen.getByText("BUTTON$CANCEL")).toBeInTheDocument();
  });

  it("should disable Launch button when no repository or branch is selected", () => {
    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    const launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
    expect(launchButton).toBeDisabled();
  });

  it("should call onClose and reset state when Cancel is clicked", async () => {
    const user = userEvent.setup();

    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    await user.click(screen.getByText("BUTTON$CANCEL"));
    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should enable Launch button after selecting repository and branch", async () => {
    const user = userEvent.setup();

    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    // Select a repository
    await user.click(screen.getByTestId("git-repo-dropdown"));

    // Select a branch
    await user.click(screen.getByTestId("git-branch-dropdown"));

    const launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
    expect(launchButton).not.toBeDisabled();
  });

  it("should call onLaunch with selected repository and branch, then close", async () => {
    const user = userEvent.setup();

    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    // Select repository and branch
    await user.click(screen.getByTestId("git-repo-dropdown"));
    await user.click(screen.getByTestId("git-branch-dropdown"));

    // Click Launch
    await user.click(screen.getByText("BUTTON$LAUNCH"));

    expect(mockOnLaunch).toHaveBeenCalledWith(
      {
        id: 1,
        full_name: "owner/repo",
        git_provider: "github",
        main_branch: "main",
      },
      { name: "main" },
    );
    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should not call onLaunch when Launch is clicked without selections", async () => {
    const user = userEvent.setup();

    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    // Force click the launch button even though it's disabled
    const launchButton = screen.getByText("BUTTON$LAUNCH").closest("button")!;
    await user.click(launchButton);

    expect(mockOnLaunch).not.toHaveBeenCalled();
  });

  it("should reset branch selection when repository changes", async () => {
    const user = userEvent.setup();

    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    // Select repository and branch
    await user.click(screen.getByTestId("git-repo-dropdown"));
    await user.click(screen.getByTestId("git-branch-dropdown"));

    // Launch button should be enabled
    let launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
    expect(launchButton).not.toBeDisabled();

    // Select a new repository (resets branch)
    await user.click(screen.getByTestId("git-repo-dropdown"));

    // Launch button should be disabled again (branch was reset)
    launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
    expect(launchButton).toBeDisabled();
  });

  it("should use small modal width", () => {
    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    // ModalBody with width="small" renders w-[384px]
    const modalBody = screen
      .getByText("CONVERSATION$OPEN_REPOSITORY")
      .closest(".bg-base-secondary");
    expect(modalBody).toHaveClass("w-[384px]");
  });

  it("should override default gap with !gap-4 for tighter spacing", () => {
    render(
      <OpenRepositoryModal
        isOpen={true}
        onClose={mockOnClose}
        onLaunch={mockOnLaunch}
      />,
    );

    const modalBody = screen
      .getByText("CONVERSATION$OPEN_REPOSITORY")
      .closest(".bg-base-secondary");
    expect(modalBody).toHaveClass("!gap-4");
  });

  describe("provider switching", () => {
    it("should not show provider dropdown when only one provider exists", () => {
      mockProviders.current = ["github"];

      render(
        <OpenRepositoryModal
          isOpen={true}
          onClose={mockOnClose}
          onLaunch={mockOnLaunch}
        />,
      );

      expect(
        screen.queryByTestId("git-provider-dropdown"),
      ).not.toBeInTheDocument();
    });

    it("should show provider dropdown when multiple providers exist", () => {
      mockProviders.current = ["github", "gitlab"];

      render(
        <OpenRepositoryModal
          isOpen={true}
          onClose={mockOnClose}
          onLaunch={mockOnLaunch}
        />,
      );

      expect(
        screen.getByTestId("git-provider-dropdown"),
      ).toBeInTheDocument();
      expect(screen.getByTestId("provider-github")).toBeInTheDocument();
      expect(screen.getByTestId("provider-gitlab")).toBeInTheDocument();
    });

    it("should reset repository and branch when provider changes", async () => {
      mockProviders.current = ["github", "gitlab"];
      const user = userEvent.setup();

      render(
        <OpenRepositoryModal
          isOpen={true}
          onClose={mockOnClose}
          onLaunch={mockOnLaunch}
        />,
      );

      // Select repo and branch
      await user.click(screen.getByTestId("git-repo-dropdown"));
      await user.click(screen.getByTestId("git-branch-dropdown"));

      // Launch should be enabled
      let launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
      expect(launchButton).not.toBeDisabled();

      // Switch provider — should reset selections
      await user.click(screen.getByTestId("provider-gitlab"));

      // Launch should be disabled (repo and branch reset)
      launchButton = screen.getByText("BUTTON$LAUNCH").closest("button");
      expect(launchButton).toBeDisabled();
    });
  });
});
