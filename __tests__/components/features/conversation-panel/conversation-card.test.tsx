import { screen, within } from "@testing-library/react";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  test,
  vi,
} from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { ConversationCard } from "#/components/features/conversation-panel/conversation-card/conversation-card";
import { clickOnEditButton } from "./utils";
import { ConversationCardActions } from "#/components/features/conversation-panel/conversation-card/conversation-card-actions";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

// We'll use the actual i18next implementation but override the translation function

// Mock the t function to return our custom translations
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          CONVERSATION$CREATED: "Created",
          CONVERSATION$AGO: "ago",
          CONVERSATION$UPDATED: "Updated",
          COMMON$NO_REPOSITORY: "No repository",
          CONVERSATION$ACP_AGENT_GENERIC: "ACP",
          CONVERSATION_PANEL$PIN_CONVERSATION: "Pin conversation",
          CONVERSATION_PANEL$UNPIN_CONVERSATION: "Unpin conversation",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackDownloadVsCodeButtonClicked: vi.fn(),
  }),
}));

describe("ConversationCard", () => {
  const onClick = vi.fn();
  const onDelete = vi.fn();
  const onChangeTitle = vi.fn();

  beforeAll(() => {
    vi.stubGlobal("window", {
      open: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      location: { origin: "http://localhost:3000" },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  it("should render the conversation card", () => {
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    const card = screen.getByTestId("conversation-card");

    within(card).getByText("Conversation 1");

    // Use a regex to match the time part since it might have whitespace
    const timeRegex = new RegExp(
      formatTimeDelta(new Date("2021-10-01T12:00:00Z")),
    );
    expect(card).toHaveTextContent(timeRegex);
  });

  it("should render the selectedRepository if available", () => {
    const { rerender } = renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    expect(
      screen.queryByTestId("conversation-card-selected-repository"),
    ).not.toBeInTheDocument();

    rerender(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={{
          selected_repository: "org/selectedRepository",
          selected_branch: "main",
          git_provider: "github",
        }}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    screen.getByTestId("conversation-card-selected-repository");
  });

  it("styles repo and branch with the same raised pill chip as tags", () => {
    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={{
          selected_repository: "org/repo",
          selected_branch: "main",
          git_provider: "github",
        }}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        showTags
        tags={{ origin: "slack" }}
      />,
    );

    const repo = screen.getByTestId("conversation-card-selected-repository");
    const branch = screen.getByTestId("conversation-card-selected-branch");
    const tag = screen.getByTestId("conversation-card-tag-chip");

    expect(repo).toHaveClass("bg-[var(--oh-surface-raised)]");
    expect(branch).toHaveClass("bg-[var(--oh-surface-raised)]");
    expect(tag).toHaveClass("bg-[var(--oh-surface-raised)]");

    // Identical pill look. The one intentional difference is flex-shrink:
    // repo and branch share a single overflow-hidden row, so they must shrink
    // (otherwise a long repo name evicts the branch chip entirely), while a
    // tag chip keeps its intrinsic width and folds behind "+N" instead.
    const pillLook = (element: HTMLElement) =>
      element.className
        .split(/\s+/)
        .filter((name) => name !== "shrink" && name !== "shrink-0")
        .sort()
        .join(" ");

    expect(pillLook(repo)).toBe(pillLook(tag));
    expect(pillLook(branch)).toBe(pillLook(tag));
    expect(repo).toHaveClass("shrink");
    expect(repo).not.toHaveClass("shrink-0");
    expect(branch).toHaveClass("shrink");
    expect(branch).not.toHaveClass("shrink-0");
    expect(tag).toHaveClass("shrink-0");
  });

  it("stacks metadata as repo/branch, then model, then tags", () => {
    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={{
          selected_repository: "org/repo",
          selected_branch: "main",
          git_provider: "github",
        }}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        llmModel="openhands/claude-opus-4-5-20251101"
        showLlmProfiles
        agentKind="openhands"
        showTags
        tags={{ origin: "slack" }}
      />,
    );

    const repo = screen.getByTestId("conversation-card-selected-repository");
    const model = screen.getByTestId("conversation-card-agent-chip");
    const tags = screen.getByTestId("conversation-card-tag-chips");

    expect(repo.compareDocumentPosition(model) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(model.compareDocumentPosition(tags) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("renders the workspace folder name when no repository is selected", () => {
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        workspaceWorkingDir="/workspace/project/agent-canvas"
      />,
    );

    expect(screen.getByText("agent-canvas")).toBeInTheDocument();
    expect(
      screen.getByTitle("/workspace/project/agent-canvas"),
    ).toBeInTheDocument();
  });

  it("styles the no-repository label with the same raised pill chip", () => {
    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        showTags
        tags={{ origin: "slack" }}
      />,
    );

    const noRepo = screen.getByTestId("conversation-card-no-repository");
    const tag = screen.getByTestId("conversation-card-tag-chip");
    expect(noRepo).toHaveTextContent("No repository");
    expect(noRepo.className).toBe(tag.className);
  });

  it("handles Windows workspace paths and falls back when the path is empty", () => {
    const { rerender } = renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        workspaceWorkingDir="C:\\Users\\me\\repo"
      />,
    );

    expect(screen.getByText("repo")).toBeInTheDocument();

    rerender(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        workspaceWorkingDir="   "
      />,
    );

    expect(screen.getByText("No repository")).toBeInTheDocument();
  });

  it("should toggle a context menu when clicking the ellipsis button", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();
    const { rerender } = renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen={false}
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    // The closed state is observable via the `data-context-menu-open` attr
    // on the conversation-card root; visual hiding is a CSS consequence.
    expect(screen.getByTestId("conversation-card")).toHaveAttribute(
      "data-context-menu-open",
      "false",
    );

    const ellipsisButton = screen.getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    expect(onContextMenuToggle).toHaveBeenCalledWith(true);

    // Simulate context menu being opened by parent
    rerender(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    screen.getByTestId("context-menu");

    await user.click(ellipsisButton);

    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
  });

  it("should call onDelete when the delete button is clicked", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const menu = screen.getByTestId("context-menu");
    const deleteButton = within(menu).getByTestId("delete-button");

    await user.click(deleteButton);

    expect(onDelete).toHaveBeenCalled();
    expect(onContextMenuToggle).toHaveBeenCalledWith(false);
  });

  test("clicking the selectedRepository should not trigger the onClick handler", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={{
          selected_repository: "org/selectedRepository",
          selected_branch: "main",
          git_provider: "github",
        }}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    const selectedRepository = screen.getByTestId(
      "conversation-card-selected-repository",
    );
    await user.click(selectedRepository);

    expect(onClick).not.toHaveBeenCalled();
  });

  test("conversation title should call onChangeTitle when changed and blurred", async () => {
    const user = userEvent.setup();
    let menuOpen = true;
    const onContextMenuToggle = vi.fn((isOpen: boolean) => {
      menuOpen = isOpen;
    });
    const { rerender } = renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        onChangeTitle={onChangeTitle}
        contextMenuOpen={menuOpen}
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    await clickOnEditButton(user);

    // Re-render with updated state
    rerender(
      <ConversationCard
        onDelete={onDelete}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        onChangeTitle={onChangeTitle}
        contextMenuOpen={menuOpen}
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const title = screen.getByTestId("conversation-card-title");

    expect(title).toBeEnabled();
    // Context menu should be closed after edit button is clicked.
    expect(screen.getByTestId("conversation-card")).toHaveAttribute(
      "data-context-menu-open",
      "false",
    );
    // expect to be focused
    expect(document.activeElement).toBe(title);

    await user.clear(title);
    await user.type(title, "New Conversation Name   ");
    await user.tab();

    expect(onChangeTitle).toHaveBeenCalledWith("New Conversation Name");
  });

  it("should not call onChange title", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    await clickOnEditButton(user);

    const title = screen.getByTestId("conversation-card-title");

    await user.clear(title);
    await user.tab();

    expect(onChangeTitle).not.toBeCalled();
  });

  test("clicking the title should trigger the onClick handler", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ConversationCard
        onClick={onClick}
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    const title = screen.getByTestId("conversation-card-title");
    await user.click(title);

    expect(onClick).toHaveBeenCalled();
  });

  test("clicking the title should not trigger the onClick handler if edit mode", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    await clickOnEditButton(user);

    const title = screen.getByTestId("conversation-card-title");
    await user.click(title);

    expect(onClick).not.toHaveBeenCalled();
  });

  test("clicking the delete button should not trigger the onClick handler", async () => {
    const user = userEvent.setup();
    const onContextMenuToggle = vi.fn();
    renderWithProviders(
      <ConversationCard
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const menu = screen.getByTestId("context-menu");
    const deleteButton = within(menu).getByTestId("delete-button");

    await user.click(deleteButton);

    expect(onClick).not.toHaveBeenCalled();
  });

  it("should not display the edit or delete options if the handler is not provided", async () => {
    const onContextMenuToggle = vi.fn();
    const { rerender } = renderWithProviders(
      <ConversationCard
        onClick={onClick}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const menu = await screen.findByTestId("context-menu");
    expect(within(menu).queryByTestId("edit-button")).toBeInTheDocument();
    expect(within(menu).queryByTestId("delete-button")).not.toBeInTheDocument();

    rerender(
      <ConversationCard
        onClick={onClick}
        onDelete={onDelete}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        contextMenuOpen
        onContextMenuToggle={onContextMenuToggle}
      />,
    );

    const newMenu = await screen.findByTestId("context-menu");
    expect(
      within(newMenu).queryByTestId("edit-button"),
    ).not.toBeInTheDocument();
    expect(within(newMenu).queryByTestId("delete-button")).toBeInTheDocument();
  });

  it("should not render the ellipsis button if there are no actions", () => {
    const { rerender } = renderWithProviders(
      <ConversationCard
        onClick={onClick}
        onDelete={onDelete}
        onChangeTitle={onChangeTitle}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    expect(screen.getByTestId("ellipsis-button")).toBeInTheDocument();

    rerender(
      <ConversationCard
        onClick={onClick}
        onDelete={onDelete}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    expect(screen.getByTestId("ellipsis-button")).toBeInTheDocument();

    rerender(
      <ConversationCard
        onClick={onClick}
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
      />,
    );

    expect(screen.queryByTestId("ellipsis-button")).not.toBeInTheDocument();
  });

  it("renders the status dot in the header when executionStatus is provided", () => {
    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        executionStatus={ExecutionStatus.RUNNING}
      />,
    );

    expect(
      screen.getByTestId("conversation-status-working"),
    ).toBeInTheDocument();
  });

  const statusTable: [ExecutionStatus, boolean][] = [
    [ExecutionStatus.RUNNING, true],
    [ExecutionStatus.IDLE, true],
    [ExecutionStatus.FINISHED, true],
    [ExecutionStatus.WAITING_FOR_CONFIRMATION, true],
    [ExecutionStatus.ERROR, false],
    [ExecutionStatus.STUCK, false],
    [ExecutionStatus.PAUSED, false],
  ];

  it.each(statusTable)(
    "should toggle stop button visibility correctly for execution status",
    (executionStatus, shouldShow) => {
      renderWithProviders(
        <ConversationCardActions
          contextMenuOpen={true}
          onContextMenuToggle={vi.fn()}
          onStop={vi.fn()}
          executionStatus={executionStatus}
        />,
      );

      const stopButton = screen.queryByTestId("stop-button");

      if (shouldShow) {
        expect(stopButton).toBeInTheDocument();
      } else {
        expect(stopButton).not.toBeInTheDocument();
      }
    },
  );

  describe("stop button label by active backend", () => {
    const cloudBackend: Backend = {
      id: "prod",
      name: "Production",
      host: "https://app.all-hands.dev",
      apiKey: "bearer-token",
      kind: "cloud",
    };

    afterEach(() => {
      __resetActiveStoreForTests();
    });

    it("uses COMMON$STOP_CONVERSATION on a local backend", () => {
      // Default active backend (no provider, no registered backends) is the
      // bundled local backend.
      renderWithProviders(
        <ConversationCardActions
          contextMenuOpen={true}
          onContextMenuToggle={vi.fn()}
          onStop={vi.fn()}
          executionStatus={ExecutionStatus.RUNNING}
        />,
      );

      expect(screen.getByTestId("stop-button")).toHaveTextContent(
        "COMMON$STOP_CONVERSATION",
      );
    });

    it("uses COMMON$CLOSE_CONVERSATION_STOP_RUNTIME on a cloud backend", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderWithProviders(
        <ActiveBackendProvider>
          <ConversationCardActions
            contextMenuOpen={true}
            onContextMenuToggle={vi.fn()}
            onStop={vi.fn()}
            executionStatus={ExecutionStatus.RUNNING}
          />
        </ActiveBackendProvider>,
      );

      expect(screen.getByTestId("stop-button")).toHaveTextContent(
        "COMMON$CLOSE_CONVERSATION_STOP_RUNTIME",
      );
    });
  });

  describe("Tag chips", () => {
    // Tag chips surface the agent-server's server-side conversation tags
    // (e.g. ``origin=slack`` stamped by an automation) and are gated by the
    // conversation panel's "Tags" toggle (``showTags``). Chip labels are
    // value-only; the full ``key: value`` lives in the chip tooltip.
    it("renders non-reserved tags as value-only chips when showTags is on", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{ origin: "slack", owner: "alice" }}
        />,
      );

      const chips = screen.getAllByTestId("conversation-card-tag-chip");
      // ``origin`` is a priority key, so it leads; remaining keys sort A–Z.
      expect(chips).toHaveLength(2);
      expect(chips[0]).toHaveTextContent("slack");
      expect(chips[0].getAttribute("title")).toMatch(/: slack$/);
      expect(chips[0].getAttribute("title")).not.toContain("origin");
      expect(chips[1]).toHaveTextContent("alice");
      expect(chips[1]).toHaveAttribute("title", "Owner: alice");
      expect(
        within(chips[0]).getByTestId("conversation-card-tag-chip-icon"),
      ).toHaveAttribute("data-tag-key", "origin");
      expect(
        within(chips[1]).getByTestId("conversation-card-tag-chip-icon"),
      ).toHaveAttribute("data-tag-key", "owner");
    });

    it("filters reserved tag keys out of the chip row", () => {
      // Reserved keys already have a first-class UI source (ACP chip, title,
      // repo/branch/workspace metadata) and must not double-render as tags.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{
            acpserver: "claude-code",
            title: "ignored title tag",
            git_provider: "github",
            repo_name: "org/repo",
            selected_branch: "main",
            archiveworkspacepath: "/workspace/project",
            origin: "review",
          }}
        />,
      );

      const chips = screen.getAllByTestId("conversation-card-tag-chip");
      expect(chips).toHaveLength(1);
      expect(chips[0]).toHaveTextContent("review");
      expect(chips[0].getAttribute("title")).toMatch(/: review$/);
      expect(chips[0].getAttribute("title")).not.toContain("origin");
    });

    it("keeps the automation name/trigger chips but hides the automation id chips", () => {
      // The automation id/run-id tags are raw UUIDs consumed by the panel's
      // automation filter — chip noise — while the human-meaningful name and
      // trigger stay visible. Like every tag chip they render value-only,
      // with the humanized ``key: value`` pair in the tooltip.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{
            automationname: "Nightly Audit",
            automationtrigger: "cron",
            automationid: "3f2b6c1e-1111-4222-8333-abcdefabcdef",
            automationrunid: "run-0001",
          }}
        />,
      );

      const chips = screen.getAllByTestId("conversation-card-tag-chip");
      expect(chips).toHaveLength(2);
      expect(chips[0]).toHaveTextContent("Nightly Audit");
      expect(chips[0]).toHaveAttribute(
        "title",
        "Automationname: Nightly Audit",
      );
      expect(chips[1]).toHaveTextContent("cron");
      expect(chips[1]).toHaveAttribute("title", "Automationtrigger: cron");
    });

    it("hides the chips when showTags is omitted", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          tags={{ origin: "slack" }}
        />,
      );

      expect(
        screen.queryByTestId("conversation-card-tag-chip"),
      ).not.toBeInTheDocument();
    });

    it("keeps chips on a single nowrap row", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{ env: "prod", origin: "slack", owner: "alice" }}
        />,
      );

      const row = screen.getByTestId("conversation-card-tag-row");
      expect(row).toHaveClass("flex-nowrap");
      expect(row).toHaveClass("overflow-hidden");
    });

    it("hard-truncates long chip values while keeping the full tooltip", () => {
      const longValue = "abcdefghijklmnopqrstuvwxyz";
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{ token: longValue }}
        />,
      );

      const chip = screen.getByTestId("conversation-card-tag-chip");
      expect(chip).toHaveTextContent("abcdefghijklm…");
      expect(chip).toHaveAttribute("title", `Token: ${longValue}`);
    });

    it("renders no chip row when every tag is reserved", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showTags
          tags={{ acpserver: "claude-code" }}
        />,
      );

      expect(
        screen.queryByTestId("conversation-card-tag-chip"),
      ).not.toBeInTheDocument();
    });
  });

  describe("Agent chip", () => {
    // The agent chip is gated by the conversation panel's "Agent / model"
    // toggle (``showLlmProfiles``) — one control for both ACP and OpenHands
    // cards. The renders below pass ``showLlmProfiles`` to exercise the chip;
    // the omitted-prop fallback is covered by the first two tests.
    it("hides the chip when showLlmProfiles is omitted for ACP", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          agentKind="acp"
          acpServer="claude-code"
          llmModel="claude-opus-4-7"
        />,
      );

      expect(
        screen.queryByTestId("conversation-card-agent-chip"),
      ).not.toBeInTheDocument();
    });

    it("hides the chip when showLlmProfiles is omitted for OpenHands", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          agentKind="openhands"
          llmModel="claude-sonnet-4"
        />,
      );

      expect(
        screen.queryByTestId("conversation-card-agent-chip"),
      ).not.toBeInTheDocument();
    });

    it("renders the brand mark + model for an ACP conversation with a model", () => {
      // PR 730 wires ``current_model_name``/``current_model_id``/configured
      // ``acp_model`` into ``llm_model`` on the adapter so ACP conversations
      // arrive at the card with a concrete model string. With the chip toggle
      // on, the chip shows the resolved Claude brand mark + that model text.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="acp"
          acpServer="claude-code"
          llmModel="raw-model-id"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      // A raw model string not in the registry passes through verbatim
      // (label resolution for known IDs is covered by the next test).
      expect(chip).toHaveTextContent("raw-model-id");
      expect(chip).toHaveAttribute("title", "Claude Code · raw-model-id");
      expect(
        within(chip).getByTestId("agent-brand-icon-claude-code"),
      ).toBeInTheDocument();
    });

    it("shows the provider's picker label for a known model ID", () => {
      // When ``llm_model`` is a registry-known ID, the chip renders the
      // human label ("Claude Opus 4.8 (1M)") instead of the raw ID — matching
      // what the Settings → Agent picker shows for the same value.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="acp"
          acpServer="claude-code"
          llmModel="opus[1m]"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("Claude Opus 4.8 (1M)");
      expect(chip).toHaveAttribute(
        "title",
        "Claude Code · Claude Opus 4.8 (1M)",
      );
    });

    it("falls back to the provider display name for an ACP conversation with no model", () => {
      // No ``llm_model`` (older agent-server, no SDK runtime fields, no
      // configured ``acp_model``) — the chip still renders for identity, with
      // the provider name as the text and the brand mark as the icon.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="acp"
          acpServer="claude-code"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("Claude Code");
      expect(chip).toHaveAttribute("title", "Claude Code");
      expect(
        within(chip).getByTestId("agent-brand-icon-claude-code"),
      ).toBeInTheDocument();
    });

    it("falls back to the generic terminal glyph when the server key is unknown", () => {
      // ``custom`` (and any future ACP server Canvas doesn't know yet) maps
      // to the fallback ``cli-generic`` icon and the generic "ACP" label.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="acp"
          acpServer="custom"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("ACP");
      expect(
        within(chip).getByTestId("agent-brand-icon-generic"),
      ).toBeInTheDocument();
    });

    it("falls back to the generic terminal glyph when the server key is null", () => {
      // ACP conversations missing the ``acpserver`` tag (older clients,
      // raw API writes) still get a chip — identity first, exact provider
      // second.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="acp"
          acpServer={null}
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("ACP");
      expect(
        within(chip).getByTestId("agent-brand-icon-generic"),
      ).toBeInTheDocument();
    });

    it("renders the OpenHands logo + model name for native conversations", () => {
      // With the chip toggle on, OpenHands native conversations show the
      // OpenHands logo + the raw ``agent.llm.model`` string. A stray
      // ``acp_server`` value on an OpenHands card must not flip the icon to
      // the Claude/Codex/Gemini brand mark.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="openhands"
          acpServer="claude-code"
          llmModel="claude-sonnet-4"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("claude-sonnet-4");
      expect(chip).toHaveAttribute("title", "claude-sonnet-4");
      expect(
        within(chip).getByTestId("agent-brand-icon-openhands"),
      ).toBeInTheDocument();
      expect(
        within(chip).queryByTestId("agent-brand-icon-claude-code"),
      ).not.toBeInTheDocument();
    });

    it("labels a free OpenHands route on native conversation chips", () => {
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="openhands"
          llmModel="openhands/glm-5.2"
        />,
      );

      const chip = screen.getByTestId("conversation-card-agent-chip");
      expect(chip).toHaveTextContent("OpenHands GLM-5.2 (free)");
      expect(chip).toHaveAttribute("title", "openhands/glm-5.2");
    });

    it("hides the chip for OpenHands conversations with no model", () => {
      // Toggle on, but no model string and no ACP server — nothing to
      // display, so the chip collapses rather than showing a bare logo.
      renderWithProviders(
        <ConversationCard
          title="Conversation 1"
          selectedRepository={null}
          lastUpdatedAt="2021-10-01T12:00:00Z"
          showLlmProfiles
          agentKind="openhands"
          llmModel={null}
        />,
      );

      expect(
        screen.queryByTestId("conversation-card-agent-chip"),
      ).not.toBeInTheDocument();
    });
  });

  it("calls onTogglePin when the pin button is clicked", async () => {
    const onTogglePin = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        conversationId="conversation-1"
        onDelete={vi.fn()}
        onTogglePin={onTogglePin}
      />,
    );

    const card = screen.getByTestId("conversation-card");
    await user.hover(card);
    await user.click(
      screen.getByTestId("conversation-pin-toggle-conversation-1"),
    );
    expect(onTogglePin).toHaveBeenCalledTimes(1);
  });

  it("keeps the pin icon visible without hover when alwaysShowPinIcon is set", () => {
    renderWithProviders(
      <ConversationCard
        title="Conversation 1"
        selectedRepository={null}
        lastUpdatedAt="2021-10-01T12:00:00Z"
        conversationId="conversation-1"
        onDelete={vi.fn()}
        onTogglePin={vi.fn()}
        isPinned
        alwaysShowPinIcon
      />,
    );

    expect(
      screen.getByTestId("conversation-pin-toggle-conversation-1"),
    ).toBeVisible();
    expect(screen.getByRole("time")).toBeInTheDocument();
  });
});
