import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { StartTaskCard } from "#/components/features/conversation-panel/start-task-card/start-task-card";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";

const task: AppConversationStartTask = {
  id: "start-task",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  status: "WAITING_FOR_SANDBOX",
  detail: "Starting",
  app_conversation_id: null,
  created_by_user_id: null,
  agent_server_url: null,
  request: {
    initial_message: {
      role: "user",
      content: [{ type: "text", text: "Explain the project structure" }],
    },
  },
};

describe("startup conversation rows", () => {
  it("uses the prompt as a normal conversation title while the sandbox starts", () => {
    renderWithProviders(<StartTaskCard task={task} />);
    expect(
      screen.getByText("Explain the project structure"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("conversation-card")).toBeInTheDocument();
    expect(screen.queryByText("Starting")).not.toBeInTheDocument();
    expect(screen.queryByText("Waiting")).not.toBeInTheDocument();
  });

  it("keeps a queued attachment prompt visible when the request has no initial message", () => {
    renderWithProviders(
      <StartTaskCard
        task={{ ...task, request: {} }}
        initialMessage="Review these files"
      />,
    );
    expect(screen.getByText("Review these files")).toBeInTheDocument();
  });

  it("preserves the conversation title on failure and uses the normal error indicator", () => {
    renderWithProviders(
      <StartTaskCard
        task={{
          ...task,
          status: "ERROR",
          detail: "Sandbox unavailable",
          request: { title: "Project review" },
        }}
      />,
    );
    expect(screen.getByText("Project review")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-status-error")).toBeInTheDocument();
    expect(screen.queryByText("Sandbox unavailable")).not.toBeInTheDocument();
  });
});
