import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";

const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mockStopConversationMutate }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel grouped folder ordering", () => {
  setupConversationPanelTest();

  describe("Conversations header folder toggle", () => {
    const renderTwoFolders = async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "alpha-chat",
            title: "Alpha Chat",
            selected_workspace: "/workspace/alpha",
          }),
          createMockConversation({
            id: "beta-chat",
            title: "Beta Chat",
            selected_workspace: "/workspace/beta",
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      await screen.findByTestId("thread-folder-ws--workspace-alpha");
      return screen.getByTestId("conversations-header-toggle");
    };

    const folderToggles = () => [
      screen.getByTestId("thread-folder-drag-ws--workspace-alpha"),
      screen.getByTestId("thread-folder-drag-ws--workspace-beta"),
    ];

    beforeEach(() => {
      useConversationPanelPreferencesStore.setState({
        groupByWorkspace: true,
        groupFolderOrder: [],
      });
    });

    it("collapses every folder when all of them are expanded", async () => {
      const user = userEvent.setup();
      const header = await renderTwoFolders();
      expect(header).toHaveAttribute("aria-expanded", "true");

      await user.click(header);

      folderToggles().forEach((toggle) => {
        expect(toggle).toHaveAttribute("aria-expanded", "false");
      });
      expect(header).toHaveAttribute("aria-expanded", "false");
    });

    it("collapses the rest when only some folders are expanded", async () => {
      const user = userEvent.setup();
      const header = await renderTwoFolders();
      const [alphaToggle] = folderToggles();

      await user.click(alphaToggle);
      expect(header).toHaveAttribute("aria-expanded", "true");

      await user.click(header);

      folderToggles().forEach((toggle) => {
        expect(toggle).toHaveAttribute("aria-expanded", "false");
      });
    });

    it("expands every folder when all of them are collapsed", async () => {
      const user = userEvent.setup();
      const header = await renderTwoFolders();

      await user.click(header);
      await user.click(header);

      folderToggles().forEach((toggle) => {
        expect(toggle).toHaveAttribute("aria-expanded", "true");
      });
      expect(header).toHaveAttribute("aria-expanded", "true");
    });

    it("stays a static label in chronological mode", async () => {
      useConversationPanelPreferencesStore.setState({
        groupByContainer: false,
        groupByWorkspace: false,
      });

      renderConversationPanel();

      const summary = await screen.findByTestId("older-conversations-summary");
      expect(
        within(summary).queryByTestId("conversations-header-toggle"),
      ).toBeNull();
      expect(summary).toHaveTextContent("SIDEBAR$CONVERSATIONS");
    });

    it("stays a static label when the grouped view has no folders", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({ items: [], next_page_id: null });

      renderConversationPanel();

      const summary = await screen.findByTestId("older-conversations-summary");
      await waitFor(() => {
        expect(
          within(summary).queryByTestId("conversations-header-toggle"),
        ).toBeNull();
      });
      expect(summary).toHaveTextContent("SIDEBAR$CONVERSATIONS");
    });
  });

  it("reorders grouped folders via drag and drop", async () => {
    useConversationPanelPreferencesStore.setState({
      groupByWorkspace: true,
      groupFolderOrder: [],
    });

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "alpha-chat",
          title: "Alpha Chat",
          selected_workspace: "/workspace/alpha",
        }),
        createMockConversation({
          id: "beta-chat",
          title: "Beta Chat",
          selected_workspace: "/workspace/beta",
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    const alphaFolder = await screen.findByTestId(
      "thread-folder-ws--workspace-alpha",
    );
    const betaFolder = screen.getByTestId("thread-folder-ws--workspace-beta");
    expect(alphaFolder.compareDocumentPosition(betaFolder)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );

    const dragHandle = screen.getByTestId(
      "thread-folder-drag-ws--workspace-alpha",
    );
    const dataTransfer = {
      effectAllowed: "move",
      dropEffect: "move",
      data: {} as Record<string, string>,
      setData(format: string, value: string) {
        this.data[format] = value;
      },
      getData(format: string) {
        return this.data[format];
      },
    };
    fireEvent.dragStart(dragHandle, { dataTransfer });
    fireEvent.dragOver(betaFolder, { dataTransfer });
    fireEvent.drop(betaFolder, { dataTransfer });

    const reorderedAlpha = screen.getByTestId(
      "thread-folder-ws--workspace-alpha",
    );
    const reorderedBeta = screen.getByTestId(
      "thread-folder-ws--workspace-beta",
    );
    expect(reorderedBeta.compareDocumentPosition(reorderedAlpha)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(
      useConversationPanelPreferencesStore.getState().groupFolderOrder,
    ).toEqual(["ws:/workspace/beta", "ws:/workspace/alpha"]);
  });

  describe("nested container + workspace grouping (#15607)", () => {
    it("renders a container header per sandbox with that container's workspace folders nested underneath", async () => {
      useConversationPanelPreferencesStore.setState({
        groupByContainer: true,
        groupByWorkspace: true,
      });
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "a1",
            title: "Alpha in Sandbox A",
            sandbox_id: "sandbox-a",
            selected_workspace: "/workspace/alpha",
          }),
          createMockConversation({
            id: "b1",
            title: "Alpha in Sandbox B",
            sandbox_id: "sandbox-b",
            selected_workspace: "/workspace/alpha",
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      const headers = await screen.findAllByTestId(
        "conversation-group-section-header",
      );
      expect(headers.map((h) => h.textContent)).toEqual(
        expect.arrayContaining([
          expect.stringContaining("sandbox-a"),
          expect.stringContaining("sandbox-b"),
        ]),
      );
      expect(headers).toHaveLength(2);

      // Both containers happen to share the workspace name "alpha", but the
      // qualified folder ids keep them as two distinct, independently
      // collapsible folders rather than one merged "alpha" folder.
      expect(
        screen.getByTestId(
          "thread-folder-container-sandbox-a-ws--workspace-alpha",
        ),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(
          "thread-folder-container-sandbox-b-ws--workspace-alpha",
        ),
      ).toBeInTheDocument();
      expect(await screen.findByText("Alpha in Sandbox A")).toBeInTheDocument();
      expect(await screen.findByText("Alpha in Sandbox B")).toBeInTheDocument();
    });

    it("scopes drag-and-drop folder reordering to its own container, leaving other containers' order untouched", async () => {
      useConversationPanelPreferencesStore.setState({
        groupByContainer: true,
        groupByWorkspace: true,
        groupFolderOrder: [],
      });
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "a1",
            title: "A1",
            sandbox_id: "sandbox-a",
            selected_workspace: "/workspace/alpha",
          }),
          createMockConversation({
            id: "a2",
            title: "A2",
            sandbox_id: "sandbox-a",
            selected_workspace: "/workspace/beta",
          }),
          createMockConversation({
            id: "b1",
            title: "B1",
            sandbox_id: "sandbox-b",
            selected_workspace: "/workspace/gamma",
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      const betaFolder = await screen.findByTestId(
        "thread-folder-container-sandbox-a-ws--workspace-beta",
      );
      const dragHandle = screen.getByTestId(
        "thread-folder-drag-container-sandbox-a-ws--workspace-alpha",
      );

      const dataTransfer = {
        effectAllowed: "move",
        dropEffect: "move",
        data: {} as Record<string, string>,
        setData(format: string, value: string) {
          this.data[format] = value;
        },
        getData(format: string) {
          return this.data[format];
        },
      };
      fireEvent.dragStart(dragHandle, { dataTransfer });
      fireEvent.dragOver(betaFolder, { dataTransfer });
      fireEvent.drop(betaFolder, { dataTransfer });

      await waitFor(() => {
        const order =
          useConversationPanelPreferencesStore.getState().groupFolderOrder;
        expect(order).toEqual([
          "container:sandbox-a|ws:/workspace/beta",
          "container:sandbox-a|ws:/workspace/alpha",
        ]);
      });
      // sandbox-b's single folder was never part of the drag and is not
      // dropped from the persisted order (it just has nothing to reorder
      // against yet, since it's the only folder in its container).
      expect(
        screen.getByTestId(
          "thread-folder-container-sandbox-b-ws--workspace-gamma",
        ),
      ).toBeInTheDocument();
    });
  });
});
