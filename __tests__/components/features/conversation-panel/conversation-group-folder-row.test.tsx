import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConversationGroupFolderRow } from "#/components/features/conversation-panel/conversation-group-folder-row";
import type { ConversationGroupLaunch } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

const SPACE_GROUP = {
  id: "ws:/workspace/alpha",
  label: "alpha",
  conversations: [],
  launch: { workingDir: "/workspace/alpha" } satisfies ConversationGroupLaunch,
};

const NONE_GROUP = {
  id: "__none_workspace",
  label: "No workspace",
  conversations: [],
  launch: {} satisfies ConversationGroupLaunch,
};

const noopDrag = {
  onDragStart: vi.fn(),
  onDragEnd: vi.fn(),
  onDragOver: vi.fn(),
  onDragLeave: vi.fn(),
  onDrop: vi.fn(),
};

function renderRow(
  group: typeof SPACE_GROUP | typeof NONE_GROUP,
  overrides: Partial<ComponentProps<typeof ConversationGroupFolderRow>> = {},
) {
  const onToggleExpanded = vi.fn();
  const onLaunchFromGroup = vi.fn();
  const onOpenSpace = vi.fn();
  const onCreateIssue = vi.fn();
  renderWithProviders(
    <ConversationGroupFolderRow
      group={group}
      expanded
      previewExpanded={false}
      isDragging={false}
      dropIndicatorPosition={null}
      animateLayout={false}
      isCreatingConversationFlow={false}
      onToggleExpanded={onToggleExpanded}
      onTogglePreviewExpanded={vi.fn()}
      onLaunchFromGroup={onLaunchFromGroup}
      onOpenSpace={overrides.onOpenSpace ?? onOpenSpace}
      onCreateIssue={overrides.onCreateIssue ?? onCreateIssue}
      renderConversationCard={() => null}
      {...noopDrag}
      {...overrides}
    />,
  );
  return { onToggleExpanded, onLaunchFromGroup, onOpenSpace, onCreateIssue };
}

describe("ConversationGroupFolderRow", () => {
  it("opens the space kanban when the header label is clicked", () => {
    const { onOpenSpace, onToggleExpanded } = renderRow(SPACE_GROUP);

    fireEvent.click(
      screen.getByTestId("thread-folder-open-ws--workspace-alpha"),
    );

    expect(onOpenSpace).toHaveBeenCalledTimes(1);
    expect(onToggleExpanded).not.toHaveBeenCalled();
  });

  it("still expands and collapses from the folder icon", () => {
    const { onToggleExpanded, onOpenSpace } = renderRow(SPACE_GROUP);

    fireEvent.click(
      screen.getByTestId("thread-folder-drag-ws--workspace-alpha"),
    );

    expect(onToggleExpanded).toHaveBeenCalledTimes(1);
    expect(onOpenSpace).not.toHaveBeenCalled();
  });

  it("offers new chat and new issue from the plus menu on a space", () => {
    const { onLaunchFromGroup, onCreateIssue } = renderRow(SPACE_GROUP);

    fireEvent.click(
      screen.getByTestId("add-conversation-to-group-ws--workspace-alpha"),
    );
    expect(onLaunchFromGroup).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByTestId("add-to-space-new-chat-ws--workspace-alpha"),
    );
    expect(onLaunchFromGroup).toHaveBeenCalledTimes(1);

    fireEvent.click(
      screen.getByTestId("add-conversation-to-group-ws--workspace-alpha"),
    );
    fireEvent.click(
      screen.getByTestId("add-to-space-new-issue-ws--workspace-alpha"),
    );
    expect(onCreateIssue).toHaveBeenCalledTimes(1);
  });

  it("starts a chat immediately from plus when the group has no kanban", () => {
    const onLaunchFromGroup = vi.fn();
    renderRow(NONE_GROUP, {
      onOpenSpace: undefined,
      onCreateIssue: undefined,
      onLaunchFromGroup,
    });

    fireEvent.click(
      screen.getByTestId("add-conversation-to-group-__none_workspace"),
    );

    expect(onLaunchFromGroup).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByTestId("add-to-space-new-issue-__none_workspace"),
    ).not.toBeInTheDocument();
  });
});
