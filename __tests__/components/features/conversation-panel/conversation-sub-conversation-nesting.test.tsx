import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { nestSubConversations } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mockStopConversationMutate }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("nestSubConversations (unit)", () => {
  it("attaches children to their parent and removes them from the top level", () => {
    const parent = createMockConversation({ id: "parent-1" });
    const child = createMockConversation({
      id: "child-1",
      parent_conversation_id: "parent-1",
    });
    const rows = nestSubConversations([parent, child], "updated");
    expect(rows).toHaveLength(1);
    expect(rows[0]?.conversation.id).toBe("parent-1");
    expect(rows[0]?.children.map((c) => c.id)).toEqual(["child-1"]);
  });

  it("keeps an orphan child (parent not in the list) as a top-level row", () => {
    const orphan = createMockConversation({
      id: "orphan",
      parent_conversation_id: "unloaded-parent",
    });
    const standalone = createMockConversation({ id: "standalone" });
    const rows = nestSubConversations([orphan, standalone], "updated");
    expect(rows.map((r) => r.conversation.id)).toEqual([
      "orphan",
      "standalone",
    ]);
    expect(rows.every((r) => r.children.length === 0)).toBe(true);
  });

  it("sorts nested children by the requested field while keeping top-level order", () => {
    const parent = createMockConversation({
      id: "p",
      created_at: "2026-01-01T12:00:00.000Z",
      updated_at: "2026-01-01T12:00:00.000Z",
    });
    const older = createMockConversation({
      id: "older-child",
      parent_conversation_id: "p",
      created_at: "2026-01-01T10:00:00.000Z",
      updated_at: "2026-01-01T10:00:00.000Z",
    });
    const newer = createMockConversation({
      id: "newer-child",
      parent_conversation_id: "p",
      created_at: "2026-01-01T11:00:00.000Z",
      updated_at: "2026-01-01T11:00:00.000Z",
    });
    const rows = nestSubConversations([newer, parent, older], "created");
    expect(rows.map((r) => r.conversation.id)).toEqual(["p"]);
    const parentRow = rows.find((r) => r.conversation.id === "p");
    expect(parentRow?.children.map((c) => c.id)).toEqual([
      "newer-child",
      "older-child",
    ]);
  });

  it("treats conversations without the field (older agent-server) as flat", () => {
    const plain = createMockConversation({ id: "plain" });
    const rows = nestSubConversations([plain], "updated");
    expect(rows).toHaveLength(1);
    expect(rows[0]?.children).toEqual([]);
  });
});

describe("ConversationPanel sub-conversation nesting (chronological)", () => {
  setupConversationPanelTest();

  beforeEach(async () => {
    const parent = createMockConversation({
      id: "parent-conv",
      title: "Parent Conversation",
    });
    const child = createMockConversation({
      id: "child-conv",
      title: "Child Conversation",
      parent_conversation_id: "parent-conv",
    });
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [parent, child],
      next_page_id: null,
    });
    renderConversationPanel();
    await screen.findByText("Parent Conversation");
  });

  it("hides the child from the flat list and shows a toggle on the parent", async () => {
    // Child is nested: it must not render as its own top-level card.
    const cards = screen.getAllByTestId("conversation-card");
    expect(cards).toHaveLength(1);

    const toggle = screen.getByTestId(
      "conversation-sub-conversations-toggle-parent-conv",
    );
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });

  it("expands to reveal the nested child row, then collapses", async () => {
    const user = userEvent.setup();
    const toggle = screen.getByTestId(
      "conversation-sub-conversations-toggle-parent-conv",
    );
    await user.click(toggle);

    const nested = await screen.findByTestId(
      "conversation-sub-conversations-parent-conv",
    );
    const row = within(nested).getByTestId("sub-conversation-row");
    expect(row).toHaveAttribute("data-conversation-id", "child-conv");
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    await user.click(toggle);
    await waitFor(() =>
      expect(
        screen.queryByTestId("conversation-sub-conversations-parent-conv"),
      ).toBeNull(),
    );
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });
});

describe("ConversationPanel sub-conversation nesting (grouped folders)", () => {
  setupConversationPanelTest();

  beforeEach(async () => {
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
          id: "parent-conv",
          title: "Parent Conversation",
          selected_workspace: "/workspace/shared",
        }),
        createMockConversation({
          id: "child-conv",
          title: "Child Conversation",
          parent_conversation_id: "parent-conv",
          selected_workspace: "/workspace/shared",
        }),
      ],
      next_page_id: null,
    });
    renderConversationPanel();
    await screen.findByText("Parent Conversation");
  });

  it("nests the child inside its workspace folder", async () => {
    const folder = await screen.findByTestId(
      "thread-folder-ws--workspace-shared",
    );
    const cards = within(folder).getAllByTestId("conversation-card");
    expect(cards).toHaveLength(1);

    const toggle = within(folder).getByTestId(
      "conversation-sub-conversations-toggle-parent-conv",
    );
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });
});
