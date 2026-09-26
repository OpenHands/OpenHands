import { fireEvent, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
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

  it("reorders grouped folders via drag and drop", async () => {
    useConversationPanelPreferencesStore.setState({
      organizeMode: "grouped",
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
});
