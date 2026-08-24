import React from "react";
import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationRunGraph } from "#/components/features/conversation-run-graph/conversation-run-graph";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";

function conv(id: string, overrides: Partial<AppConversation> = {}) {
  return {
    id,
    title: `Title ${id}`,
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    updated_at: "2026-01-01T00:00:00Z",
    created_at: "2026-01-01T00:00:00Z",
    execution_status: ExecutionStatus.FINISHED,
    conversation_url: null,
    metrics: null,
    llm_model: null,
    trigger: null,
    sub_conversation_ids: [],
    tags: {},
    ...overrides,
  } as unknown as AppConversation;
}

describe("ConversationRunGraph", () => {
  it("renders the parent and every child as clickable nodes", async () => {
    renderWithProviders(
      <ConversationRunGraph
        parent={conv("p", { title: "Plan conversation" })}
        childConversations={[conv("c1"), conv("c2")]}
      />,
    );

    expect(screen.getByTestId("run-graph-parent-node")).toHaveTextContent(
      "Plan conversation",
    );
    const childNodes = screen.getAllByTestId("run-graph-child-node");
    expect(childNodes).toHaveLength(2);
    expect(childNodes[0]).toHaveTextContent("Title c1");
    expect(childNodes[1]).toHaveTextContent("Title c2");
  });

  it("opens the normal conversation view when a node is clicked", async () => {
    const user = userEvent.setup();
    const navigate = vi.fn();
    renderWithProviders(
      <ConversationRunGraph
        parent={conv("p")}
        childConversations={[conv("c1"), conv("c2")]}
      />,
      { navigation: { navigate } },
    );

    await user.click(screen.getByTestId("run-graph-parent-node"));
    expect(navigate).toHaveBeenCalledWith("/conversations/p", {
      replace: false,
    });

    navigate.mockClear();
    await user.click(screen.getAllByTestId("run-graph-child-node")[1]);
    expect(navigate).toHaveBeenCalledWith("/conversations/c2", {
      replace: false,
    });
  });
});
