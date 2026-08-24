import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { ConversationNode } from "#/components/features/conversation-panel/conversation-panel-list-helpers";
import {
  ConversationTree,
  CONVERSATION_TREE_INDENT_PX,
} from "#/components/features/conversation-panel/conversation-tree";

function node(
  id: string,
  depth = 0,
  children: ConversationNode[] = [],
): ConversationNode {
  const conversation = {
    id,
    title: `conv-${id}`,
  } as AppConversation;
  return { conversation, depth, children, hasChildren: children.length > 0 };
}

const card = (c: AppConversation) => (
  <div data-testid={`card-${c.id}`}>{c.id}</div>
);

describe("ConversationTree", () => {
  it("renders every node and indents children by one layer", () => {
    const a = node("a", 0, [node("b", 1), node("c", 1)]);
    render(
      <ConversationTree
        nodes={[a]}
        collapsedIds={new Set()}
        onToggleCollapsed={vi.fn()}
        renderConversationCard={card}
      />,
    );

    expect(screen.getByTestId("card-a")).toBeTruthy();
    expect(screen.getByTestId("card-b")).toBeTruthy();
    expect(screen.getByTestId("card-c")).toBeTruthy();

    expect(
      screen.getByTestId("conversation-tree-row-a").style.paddingLeft,
    ).toBe("0px");
    expect(
      screen.getByTestId("conversation-tree-row-b").style.paddingLeft,
    ).toBe(`${CONVERSATION_TREE_INDENT_PX}px`);
  });

  it("collapses and expands a layer via its toggle", () => {
    const toggle = vi.fn();
    const a = node("a", 0, [node("b", 1)]);
    const { rerender } = render(
      <ConversationTree
        nodes={[a]}
        collapsedIds={new Set()}
        onToggleCollapsed={toggle}
        renderConversationCard={card}
      />,
    );
    expect(screen.getByTestId("card-b")).toBeTruthy();

    fireEvent.click(screen.getByTestId("conversation-tree-toggle-a"));
    expect(toggle).toHaveBeenCalledWith("a");

    // Collapsed: child hidden.
    rerender(
      <ConversationTree
        nodes={[a]}
        collapsedIds={new Set(["a"])}
        onToggleCollapsed={toggle}
        renderConversationCard={card}
      />,
    );
    expect(screen.queryByTestId("card-b")).toBeNull();
  });

  it("disables the toggle for leaves", () => {
    render(
      <ConversationTree
        nodes={[node("leaf", 0)]}
        collapsedIds={new Set()}
        onToggleCollapsed={vi.fn()}
        renderConversationCard={card}
      />,
    );
    expect(screen.getByTestId("conversation-tree-toggle-leaf")).toHaveAttribute(
      "disabled",
    );
  });

  it("renders nothing for an empty forest", () => {
    const { container } = render(
      <ConversationTree
        nodes={[]}
        collapsedIds={new Set()}
        onToggleCollapsed={vi.fn()}
        renderConversationCard={card}
      />,
    );
    expect(container.firstChild).toBeNull();
  });
});
