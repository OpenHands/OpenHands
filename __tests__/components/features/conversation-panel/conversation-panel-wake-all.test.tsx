import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

const mocks = vi.hoisted(() => ({
  runConversation: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual =
    await vi.importActual<typeof import("@openhands/typescript-client/clients")>(
      "@openhands/typescript-client/clients",
    );
  return {
    ...actual,
    ConversationClient: vi.fn(function ConversationClientMock() {
      return { runConversation: mocks.runConversation };
    }),
  };
});

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

const seedList = (rows: AppConversation[]) => {
  vi.spyOn(
    AgentServerConversationService,
    "searchConversations",
  ).mockResolvedValue({ items: rows, next_page_id: null });
  vi.spyOn(
    AgentServerConversationService,
    "batchGetAppConversations",
  ).mockImplementation(async (ids) =>
    ids.map((id) => rows.find((row) => row.id === id) ?? null),
  );
};

describe("ConversationPanel wake all agents", () => {
  setupConversationPanelTest();

  it("hides the wake-all button while nothing is asleep", async () => {
    seedList([
      createMockConversation({
        id: "1",
        title: "Running Project",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Finished Project",
        execution_status: ExecutionStatus.FINISHED,
      }),
    ]);

    renderConversationPanel();

    expect(await screen.findAllByTestId("conversation-card")).toHaveLength(2);
    expect(screen.queryByTestId("wake-all-agents")).not.toBeInTheDocument();
  });

  it("wakes every sleeping conversation at once, skipping archived and gone runtimes", async () => {
    const user = userEvent.setup();
    seedList([
      createMockConversation({
        id: "1",
        title: "Running Project",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Stopped Project",
        execution_status: ExecutionStatus.PAUSED,
        conversation_url: "https://runtime.example.com/api/conversations/2",
      }),
      createMockConversation({
        id: "3",
        title: "Paused Sandbox Project",
        execution_status: ExecutionStatus.IDLE,
        sandbox_id: "sandbox-3",
        sandbox_status: "PAUSED",
        conversation_url: "https://runtime.example.com/api/conversations/3",
      }),
      // A gone runtime cannot be woken, and an archived conversation should
      // not be brought back to billing by a bulk wake.
      createMockConversation({
        id: "4",
        title: "Gone Runtime Project",
        execution_status: ExecutionStatus.PAUSED,
        sandbox_status: "MISSING",
      }),
      createMockConversation({
        id: "5",
        title: "Archived Stopped Project",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ]);
    useArchivedConversationsStore
      .getState()
      .archiveConversation("default-local", "5");
    mocks.runConversation.mockResolvedValue({ success: true });

    renderConversationPanel();

    await screen.findAllByTestId("conversation-card");
    await user.click(screen.getByTestId("wake-all-agents"));

    await waitFor(() => {
      expect(mocks.runConversation).toHaveBeenCalledTimes(2);
    });
    expect(mocks.runConversation).toHaveBeenCalledWith("2");
    expect(mocks.runConversation).toHaveBeenCalledWith("3");
    expect(vi.mocked(displaySuccessToast)).toHaveBeenCalledWith(
      "TOAST$ALL_AGENTS_WOKEN",
    );
    expect(vi.mocked(displayErrorToast)).not.toHaveBeenCalled();
  });

  it("reports a failed wake as a count instead of hiding it", async () => {
    const user = userEvent.setup();
    seedList([
      createMockConversation({
        id: "2",
        title: "Stopped Project",
        execution_status: ExecutionStatus.PAUSED,
        conversation_url: "https://runtime.example.com/api/conversations/2",
      }),
      createMockConversation({
        id: "3",
        title: "Other Stopped Project",
        execution_status: ExecutionStatus.PAUSED,
        conversation_url: "https://runtime.example.com/api/conversations/3",
      }),
    ]);
    mocks.runConversation.mockImplementation(async (id: string) => {
      if (id === "3") {
        throw new Error("runtime unreachable");
      }
      return { success: true };
    });

    renderConversationPanel();

    await screen.findAllByTestId("conversation-card");
    await user.click(screen.getByTestId("wake-all-agents"));

    await waitFor(() => {
      expect(vi.mocked(displayErrorToast)).toHaveBeenCalledWith(
        "TOAST$FAILED_TO_WAKE_ALL_AGENTS",
      );
    });
    expect(mocks.runConversation).toHaveBeenCalledTimes(2);
    expect(vi.mocked(displaySuccessToast)).not.toHaveBeenCalled();
  });
});
