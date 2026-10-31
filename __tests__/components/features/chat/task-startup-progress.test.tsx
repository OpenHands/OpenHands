import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import { TaskStartupProgress } from "#/components/features/chat/task-startup-progress";
import { NavigationProvider } from "#/context/navigation-context";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationStore } from "#/stores/conversation-store";
import {
  setPendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import { restoreStartupAttachmentDraft } from "#/utils/restore-startup-attachment-draft";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { getStartTask: vi.fn(), createConversation: vi.fn() } }),
);
const navigate = vi.fn();
let client: QueryClient;
const task = {
  id: "123",
  status: "WORKING",
  request: {},
  app_conversation_id: null,
};
const file = new File(["notes"], "notes.txt");
function queue() {
  setPendingTaskAttachments("123", {
    content: "Read my notes",
    files: [file],
    images: [],
    imagesMarkedUploadAsFile: [],
  });
}
function mount() {
  return render(
    <QueryClientProvider client={client}>
      <NavigationProvider
        value={{
          navigate,
          currentPath: "/conversations/task-123",
          conversationId: "task-123",
          isNavigating: false,
        }}
      >
        <TaskStartupProgress />
      </NavigationProvider>
    </QueryClientProvider>,
  );
}
beforeEach(() => {
  vi.clearAllMocks();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  usePendingTaskAttachmentsStore.setState({ byTaskId: {} });
  useConversationStore.getState().resetConversationState();
  vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
    task as never,
  );
});
afterEach(() => client.clear());

describe("startup error recovery", () => {
  it("retries a status read without creating another conversation", async () => {
    vi.mocked(
      AgentServerConversationService.getStartTask,
    ).mockRejectedValueOnce(
      new HttpError(403, "Forbidden", { detail: "Status unavailable" }),
    );
    mount();
    await screen.findByRole("alert");
    fireEvent.click(
      screen.getByRole("button", { name: "CHAT_INTERFACE$MESSAGE_RETRY" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    );
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      2,
    );
    expect(
      AgentServerConversationService.createConversation,
    ).not.toHaveBeenCalled();
  });
  it("offers a status retry when the task is missing instead of spinning forever", async () => {
    vi.mocked(
      AgentServerConversationService.getStartTask,
    ).mockResolvedValueOnce(null);
    mount();
    await screen.findByRole("alert");
    fireEvent.click(
      screen.getByRole("button", { name: "CHAT_INTERFACE$MESSAGE_RETRY" }),
    );
    await waitFor(() =>
      expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
        2,
      ),
    );
    expect(
      AgentServerConversationService.createConversation,
    ).not.toHaveBeenCalled();
  });

  it("returns a terminally failed start's prompt and files for editing", async () => {
    queue();
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue({
      ...task,
      status: "ERROR",
      detail: "Sandbox could not start",
    } as never);
    mount();
    await screen.findByText("Sandbox could not start");
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$EDIT" }));
    expect(navigate).toHaveBeenCalledWith("/conversations");
    expect(useConversationStore.getState().messageToSend).toMatchObject({
      text: "Read my notes",
      target: "home",
    });
    expect(useConversationStore.getState().files).toEqual([file]);
  });
  it("opens an existing conversation after upload failure and restores the draft once", async () => {
    queue();
    setPendingTaskAttachments("123", {
      ...usePendingTaskAttachmentsStore.getState().byTaskId["123"],
      error: "Upload interrupted",
      conversationId: "existing",
    });
    mount();
    fireEvent.click(
      screen.getByRole("button", { name: "CONVERSATION$OPEN_CREATED" }),
    );
    expect(navigate).toHaveBeenCalledWith("/conversations/existing");
    useConversationStore.getState().resetConversationState();
    restoreStartupAttachmentDraft("existing");
    restoreStartupAttachmentDraft("existing");
    expect(useConversationStore.getState().files).toEqual([file]);
    expect(useConversationStore.getState().messageToSend).toMatchObject({
      text: "Read my notes",
      target: "conversation",
    });
    expect(
      AgentServerConversationService.createConversation,
    ).not.toHaveBeenCalled();
  });
});
