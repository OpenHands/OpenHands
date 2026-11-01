import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PendingTaskAttachmentDelivery } from "#/components/features/chat/pending-task-attachment-delivery";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  setPendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { sendMessageWithAttachments } from "#/utils/send-message-with-attachments";

vi.mock("#/utils/send-message-with-attachments", () => ({
  sendMessageWithAttachments: vi.fn(),
}));
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { getStartTask: vi.fn() } }),
);
vi.mock("#/i18n", () => ({
  default: { getFixedT: () => (key: string) => key, t: (key: string) => key },
  OPENHANDS_I18N_NAMESPACE: "openhands",
  waitForI18n: vi.fn().mockResolvedValue(undefined),
}));
const client = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});
const file = new File(["notes"], "notes.txt");
const ready = {
  id: "task",
  status: "READY",
  app_conversation_id: "conversation",
  request: {},
};

beforeEach(() => {
  vi.clearAllMocks();
  setRegisteredBackends([
    {
      id: "cloud",
      name: "Cloud",
      host: "https://example.test",
      apiKey: "",
      kind: "cloud",
    },
  ]);
  setActiveSelection({ backendId: "cloud", orgId: "org-1" });
  usePendingTaskAttachmentsStore.setState({ byTaskId: {} });
  setPendingTaskAttachments("task", {
    backendId: "cloud",
    orgId: "org-1",
    content: "Read notes",
    files: [file],
    images: [],
    imagesMarkedUploadAsFile: [],
  });
  vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
    ready as never,
  );
  vi.mocked(sendMessageWithAttachments).mockResolvedValue({} as never);
});
afterEach(() => {
  client.clear();
  setRegisteredBackends([]);
});
function mount() {
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <PendingTaskAttachmentDelivery />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("initial attachments while visiting another page", () => {
  it("delivers the initial message without a mounted conversation page", async () => {
    mount();
    await waitFor(() =>
      expect(sendMessageWithAttachments).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({
          conversationId: "conversation",
          content: "Read notes",
          files: [file],
          requireAllAttachments: true,
        }),
      ),
    );
    expect(
      usePendingTaskAttachmentsStore.getState().byTaskId.task,
    ).toBeUndefined();
  });

  it("retains an upload failure without automatically resending", async () => {
    vi.mocked(sendMessageWithAttachments).mockRejectedValue(
      new Error("Upload interrupted"),
    );
    mount();
    await waitFor(() =>
      expect(
        usePendingTaskAttachmentsStore.getState().byTaskId.task,
      ).toMatchObject({
        content: "Read notes",
        files: [file],
        error: "Upload interrupted",
      }),
    );
    expect(sendMessageWithAttachments).toHaveBeenCalledTimes(1);
  });

  it("does not read or deliver another organization's pending start", async () => {
    setActiveSelection({ backendId: "cloud", orgId: "org-2" });
    mount();
    expect(AgentServerConversationService.getStartTask).not.toHaveBeenCalled();
    expect(sendMessageWithAttachments).not.toHaveBeenCalled();
    expect(
      usePendingTaskAttachmentsStore.getState().byTaskId.task.files,
    ).toEqual([file]);
  });
});
