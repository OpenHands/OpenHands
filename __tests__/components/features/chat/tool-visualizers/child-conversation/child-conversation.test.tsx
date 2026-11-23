import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { childConversationVisualizer } from "#/components/features/chat/tool-visualizers/child-conversation/child-conversation";
import {
  renderVisualizer,
  startChildConversationAction,
  startChildConversationObservation,
} from "../test-utils";

const Body = childConversationVisualizer.Body;

describe("childConversationVisualizer", () => {
  it("shows the brief while the launch is still pending", () => {
    renderVisualizer(
      <Body
        action={startChildConversationAction({ task: "Port the parser" })}
      />,
    );
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TASK_LABEL"),
    ).toBeInTheDocument();
    expect(screen.getByText("Port the parser")).toBeInTheDocument();
    expect(
      screen.queryByTestId("child-conversation-link"),
    ).not.toBeInTheDocument();
  });

  it("links to the conversation the server started", () => {
    renderVisualizer(
      <Body
        action={startChildConversationAction()}
        observation={startChildConversationObservation({
          conversation_id: "abc123",
        })}
      />,
    );
    expect(screen.getByText("COMMON$STATUS")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(
      screen.getByText("CHILD_CONVERSATION$ISOLATION"),
    ).toBeInTheDocument();
    expect(screen.getByText("worktree")).toBeInTheDocument();
    expect(
      screen.getByText("CONVERSATION$OVERVIEW_WORKSPACE"),
    ).toBeInTheDocument();
    expect(screen.getByText("/workspaces/child-1")).toBeInTheDocument();
    expect(screen.getByTestId("child-conversation-link")).toHaveAttribute(
      "href",
      "/conversations/abc123",
    );
  });

  it("falls back to the server-provided url when it has no conversation id", () => {
    renderVisualizer(
      <Body
        observation={startChildConversationObservation({
          conversation_id: null,
          url: "https://example.test/conversations/def456",
        })}
      />,
    );
    expect(screen.getByTestId("child-conversation-link")).toHaveAttribute(
      "href",
      "https://example.test/conversations/def456",
    );
  });

  it("shows a failed launch as an error, without a link", () => {
    renderVisualizer(
      <Body
        observation={startChildConversationObservation({
          is_error: true,
          status: null,
          content: [{ type: "text", text: "Workspace has no git repository." }],
        })}
      />,
    );
    expect(
      screen.getByText("Workspace has no git repository."),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("child-conversation-link"),
    ).not.toBeInTheDocument();
  });
});
